// Last revision: 2021.4.21
// rule confidence is: 0.7	change the confidence threshold: rule_path
//transd
#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include<cstdlib>
#include<sstream>
#include<omp.h>
using namespace std;


#define pi 3.1415926535897932384626433832795


map<vector<int>,string> path2s;  // path convert to string


map<pair<string,int>,double>  path_confidence;

bool L1_flag=1;

//normal distribution 返回min 与max之间的一个数
double rand(double min, double max)
{
    return min+(max-min)*rand()/(RAND_MAX+1.0); // RAND_MAX 是 rand 所能返回的最大数值
}

//正太分布函数
double normal(double x, double miu,double sigma)
{
    return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

double randn(double miu,double sigma, double min ,double max)
{
    double x,y,dScope;
    do{
        x=rand(min,max);
        y=normal(x,miu,sigma);
        dScope=rand(0.0,normal(miu,miu,sigma));
    }while(dScope>y);
    return x;
}
// 平方函数
double sqr(double x)
{
    return x*x;
}

// calculate the length of the vector （模）
double vec_len(vector<double> &a)
{
	double res=0;
	for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	res = sqrt(res);
	return res;
}

string version;
//定义数组
char buf[100000],buf1[100000],buf2[100000];
//实体数量，关系数量
int relation_num,entity_num;
map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
map<pair<int, int>, pair<int, double>> rule2rel;		// used for path compositon by R2 rules
map<int, vector<pair<int, double> > > rel2rel;			// used for relations association by R1 rules
map<pair<int, int>, int> rule_ok;

// 两步关系路径
vector<vector<pair<int,int> > > path;

class Train{

public:
	map<pair<int,int>, map<int,int> > ok;
    void add(int x,int y,int z, vector<pair<vector<int>,double> > path_list)
    {
	// add head entity: x, tail entity: y, relation: z, relation path: path_list, ok: 1 if the triple x-z-y added
        fb_h.push_back(x);
        fb_r.push_back(z);
        fb_l.push_back(y);
	    fb_path.push_back(path_list);
        ok[make_pair(x,z)][y]=1;
    }
    void pop()
    {
        fb_h.pop_back();
        fb_r.pop_back();
        fb_l.pop_back();
        fb_path.pop_back();
    }
    void run()
    {  
		// 嵌入维度
        n = 100;
        rate = 0.001;
        //正则化系数，好像没用到
        regul = 0.01;
        cout<<"n="<<n<<' '<<"rate="<<rate<<endl;
        //定义关系矩阵
        relation_vec.resize(relation_num);
		for (int i=0; i<relation_vec.size(); i++)
			relation_vec[i].resize(n);
		//定义实体矩阵
        entity_vec.resize(entity_num);
		for (int i=0; i<entity_vec.size(); i++)
			entity_vec[i].resize(n);
		//
        relation_tmp.resize(relation_num);
		for (int i=0; i<relation_tmp.size(); i++)
			relation_tmp[i].resize(n);
		//
        entity_tmp.resize(entity_num);
		for (int i=0; i<entity_tmp.size(); i++)
			entity_tmp[i].resize(n);
		//对关系矩阵进行初始化正态分布
        for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<n; ii++)            
                relation_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
			// norm(relation_vec[i]);
        }
		// FILE* f1 = fopen("./data_FB15K/relation2vec.txt","r");
        // for (int i=0; i<relation_num; i++)
        // {
        //     for (int ii=0; ii<n; ii++)
        //     	fscanf(f1,"%lf",&relation_vec[i][ii]);
        // }
        // fclose(f1);
		//对实体矩阵进行初始化正太分布
        for (int i=0; i<entity_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                entity_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
            norm(entity_vec[i]);
        }
		// FILE* f2 = fopen("./data_FB15K/entity2vec.txt","r");
        // for (int i=0; i<entity_num; i++)
        // {
        //     for (int ii=0; ii<n; ii++)
        //     	fscanf(f1,"%lf",&entity_vec[i][ii]);
        //     norm(entity_vec[i]);
        // }
        // fclose(f2);
        bfgs();
    }

private:
    int n;
    double res;//loss function value
    double count,count1;//loss function gradient
    double rate;//learning rate
    double belta; //rule1 confidence
    double regul; //regulation factor
    int relrules_used; // relation pair rule R1 num
    vector<int> fb_h,fb_l,fb_r;  // ID of the head entity, tail entity and relation
    vector<vector<pair<vector<int>,double> > >fb_path;   // all the relation paths
    vector<vector<double> > relation_vec,entity_vec;   // entity and relation embeddings to be learned （关系实体矩阵）
    vector<vector<double> > relation_tmp,entity_tmp;//
    // vector<vector<vector<double> > > A, A_tmp,B,B_tmp;
	vector<vector<vector<double> > > R, R_tmp;

	//单位化
    void norm(vector<double> &a)
    {   
		//计算向量的模
        double x = vec_len(a);
        if (x>1)
        for (int ii=0; ii<a.size(); ii++)
                a[ii]/=x;
    }
    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        while (res<0)
            res+=x;
        return res;
    }

    void bfgs()
    {
	// training procedure
        double margin = 1,margin_rel = 1;
        cout<<"margin="<<' '<<margin<<"margin_rel="<<margin_rel<<endl;
        res=0;
        int nbatches=100;
        int nepoches =500;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         00;
        cout<<"nbatches: "<<nbatches<<"\n";
        cout<<"nepoches: "<<nepoches<<"\n";
		//batch size
        int batchsize = fb_h.size()/nbatches;
        cout<<"The total number of triples is: "<<fb_h.size()<<"\n";
        cout<<"batchsize is: "<<batchsize<<"\n";
        relation_tmp=relation_vec;
        entity_tmp = entity_vec;
	    //epoch process
        for (int epoch=0; epoch<nepoches; epoch++)
        {
			//loss 
        	res=0;
        	// relation rule R2 num
            int rules_used = 0;
            // relation pair rule R1 num
            relrules_used = 0;
         	for (int batch = 0; batch<nbatches; batch++)
         	{
			// random select one entity
			int e1 = rand_max(entity_num);
         		for (int k=0; k<batchsize; k++)
         		{
					// random select a negative entity 
					int entity_neg=rand_max(entity_num);
					int i=rand_max(fb_h.size());
					//选择一个三元组
					int e1 = fb_h[i], rel = fb_r[i], e2  = fb_l[i];
					//返回一个0-99的数
					int rand_tmp = rand()%100;
					//负采样尾部实体
					if (rand_tmp<25)
					{   
						//替换尾部实体，直到这个三元组不存在
						while (ok[make_pair(e1,rel)].count(entity_neg)>0)
							entity_neg=rand_max(entity_num);
                        train_kb(e1,e2,rel,e1,entity_neg,rel,margin);
					}
					//负采样头部实体
					else if (rand_tmp<50)
					{
						// 替换头部实体，直到这个三元组不存在
						while (ok[make_pair(entity_neg,rel)].count(e2)>0)
							entity_neg=rand_max(entity_num);
				        train_kb(e1,e2,rel,entity_neg,e2,rel,margin);
					}
					//负采样关系
					else
					{
						int rel_neg = rand_max(relation_num);
						//替换关系，直到这个三元组不存在
						while (ok[make_pair(e1,rel_neg)].count(e2)>0)
							rel_neg = rand_max(relation_num);
				        train_kb(e1,e2,rel,e1,e2,rel_neg,margin);
					}
					//三元组（h,r,t）之间存在路径
					if (fb_path[i].size()>0)
					{
						// the training procedure of paths
						int rel_neg = rand_max(relation_num);
						//负采样路径导出的关系
						while (ok[make_pair(e1,rel_neg)].count(e2)>0)
							rel_neg = rand_max(relation_num);
						//遍历每一条关系路径
						for (int path_id = 0; path_id<fb_path[i].size(); path_id++)
						{
						    // 关系路径上的关系ID列表
							vector<int> rel_path = fb_path[i][path_id].first;
							string  s = "";
							if (path2s.count(rel_path)==0)
							{   
								//创建一个输出
							    ostringstream oss;
								for (int ii=0; ii<rel_path.size(); ii++)
								{
									oss<<rel_path[ii]<<" ";
								}
								//将string类型的oss写给s
							    s=oss.str();
							    // pair <vector<int> , string>
								path2s[rel_path] = s;
							}
							//拿出路径id列表对应的路径string
							s = path2s[rel_path];
                            // the reliability of the path (路径置信度 )
							double pr = fb_path[i][path_id].second;
							// 
							double pr_path = 0;
							//规则头
							int rel_integ;
							// 引导规则头规则置信度
							double confi_integ = 0;
							//初始化规则置信度
							double confi_path = 1;
							// map<pair<string,int>,double>
							if (path_confidence.count(make_pair(s,rel))>0)
							    //关系路径置信度
								pr_path = path_confidence[make_pair(s,rel)];
							// 为了防止路径推不出特定的关系，导致pr_path为0
							pr_path = 0.99*pr_path + 0.01;  
							//关系路径上的关系条数大于1，也就是两跳
							if (rel_path.size() > 1){
							    // for (int i = 0; i < rel_path.size(); i++){
									//map<pair<int, int>, pair<int, double>>
							        if (rule2rel.count(make_pair(rel_path[0], rel_path[2])) > 0){
								    rules_used++;  // the amount of rules R2 used
								    // 规则头
                                    rel_integ = rule2rel[make_pair(rel_path[0], rel_path[2])].first;
                                    // 规则置信度
                                    confi_integ = rule2rel[make_pair(rel_path[0], rel_path[2])].second;
								    // 指导规则的置信度的乘积(miu)
								    confi_path = confi_path * confi_integ;
								    //规则头
								    rel_path[0] = rel_integ;
								    // for (int j = (i+1); j < (rel_path.size() - 1); j++){
								    //     rel_path[j] = rel_path[j+1];
								    // }
								    // 移除关系路径上的关系和实体 
								    rel_path.pop_back();
									rel_path.pop_back();
									train_path(rel, rel_neg, rel_path, margin, pr * pr_path * confi_path);
                                    }else{
										//规则无法合并，导出,这里暂且不处理
										rel_path[1] = rel_path[2];
										rel_path.pop_back();
										// 
										train_path(rel, rel_neg, rel_path, margin, pr * pr_path);
									}							   
							}else{
								// 关系路径上的关系条数为1
								train_path(rel, rel_neg, rel_path, margin, pr * pr_path);
							}								
						}
					}

					// 单位化
                    norm(relation_tmp[rel]);
            		norm(entity_tmp[e1]);
            		norm(entity_tmp[e2]);
            		norm(entity_tmp[entity_neg]);
			        e1 = e2;
 
         		}
				// 更新实体，关系的嵌入向量,
	            relation_vec = relation_tmp;
	            entity_vec = entity_tmp;
         	}
            cout<<"epoch:"<<epoch<<' '<<res<<endl;
	        cout<<"The number of R2 rules (rules of length 2) used in this epoch is: "<<rules_used<<"\n";
	        cout<<"The number of R1 rules (rules of length 1) used in this epoch is: "<<relrules_used<<"\n";
	    if (epoch>400 && (epoch+1)%100==0){
			    //500个epoch 
                int save_n = (epoch+1)/100;
				// 5
                string serial = to_string(save_n);
                FILE* f2 = fopen(("./data_FB15K/res/relation2vec_rule70_"+serial+".txt").c_str(),"w");
                FILE* f3 = fopen(("./data_FB15K/res/entity2vec_rule70_"+serial+".txt").c_str(),"w");
                for (int i=0; i<relation_num; i++)
                {
                    for (int ii=0; ii<n; ii++)
                        fprintf(f2,"%.6lf\t",relation_vec[i][ii]);
                    fprintf(f2,"\n");
                }
                for (int i=0; i<entity_num; i++)
                {
                    for (int ii=0; ii<n; ii++)
                        fprintf(f3,"%.6lf\t",entity_vec[i][ii]);
                    fprintf(f3,"\n");
                }
                fclose(f2);
                fclose(f3);
                cout<<"Saving the training result succeed!"<<endl;
                }

	    }  // epoch
    }   // bfgs()

    double res1;

    // calculate the direct triple along with the typical translation-based methods
    double calc_kb(int e1,int e2,int rel)
   {
       double sum=0;
       for (int ii=0; ii<n; ii++)
		{
		    // t-h-r
			double tmp = entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii];
	        if (L1_flag)
				sum+=fabs(tmp);
			else
				sum+=sqr(tmp);
		}
		if(L1_flag)
          return sum;
       else
          return sqrt(sum);
    }

    // calculate the similarity between two relations
    double calc_rule(int rel, int relpn){
	double sum = 0;
	for (int ii = 0; ii < n; ii++){
	   // ||r− re||
		double tmp = relation_vec[rel][ii] - relation_vec[relpn][ii];
		if (L1_flag)
			sum += fabs(tmp);
		else
			sum += sqr(tmp);
	}
        if(L1_flag)
           return sum;
        else
           return sqrt(sum);
    }

    void gradient_kb(int e1,int e2,int rel, double belta)
    {
       for (int ii=0; ii<n; ii++)
       {
           //计算梯度 t-h -r
           double x = 2*(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
           // L1
           if (L1_flag)
           	if (x>0)
           		x=1;
           	else
           		x=-1;
           relation_tmp[rel][ii]-=belta*rate*x;
           entity_tmp[e1][ii]-=belta*rate*x;
           entity_tmp[e2][ii]+=belta*rate*x;
       }
    }

    // gradient of relation association
    void gradient_rule(int rel1, int rel2, double belta)
    {
	for (int ii=0; ii<n; ii++){
	    //计算梯度
		double x = 2*(relation_vec[rel1][ii] - relation_vec[rel2][ii]);
		// LOSS采用L1
		if (L1_flag)
			if (x>0)
				x = 1;
			else
				x = -1;
		// 正样本rel1越小，rel2越大，负样本相反
		relation_tmp[rel1][ii] += belta*rate*x;
		relation_tmp[rel2][ii] -= belta*rate*x;
	    }
    }

    // 路径处理 Cp - r
    double calc_path(int r1,vector<int> rel_path)
    {
    // calculate the similarity between path and relation pair
        double sum=0;
        for (int ii=0; ii<n; ii++)
		{
		// 直接关系r
			double tmp = relation_vec[r1][ii];
		//直接关系减去关系路径上的关系
			for (int j=0; j<rel_path.size(); j++)
				tmp-=relation_vec[rel_path[j]][ii];
	        if (L1_flag)
				sum+=fabs(tmp);
			else
				sum+=sqr(tmp);
		}
        if(L1_flag)
           return sum;
        else
           return sqrt(sum);
    }
    void gradient_path(int r1,vector<int> rel_path, double belta)
    {
        for (int ii=0; ii<n; ii++)
        {

			double x = relation_vec[r1][ii];
			for (int j=0; j<rel_path.size(); j++)
				x-=relation_vec[rel_path[j]][ii];
            if (L1_flag)
            	if (x>0)
            		x=1;
            	else
            		x=-1;
            relation_tmp[r1][ii]+=belta*rate*x;
			for (int j=0; j<rel_path.size(); j++)
            	relation_tmp[rel_path[j]][ii]-=belta*rate*x;

        }
    }
    void train_kb(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,double margin)
    {
        double sum1 = calc_kb(e1_a,e2_a,rel_a);
        double sum2 = calc_kb(e1_b,e2_b,rel_b);
        // weight of length 1 rules in loss function
	    double lambda_rule = 3;
		//weight of paths and length 2 rules in loss function
	    double marginrule = 1;

	    // L1计算
        if (sum1+margin>sum2)
        {
        	res+=margin+sum1-sum2;
        	// 梯度更新正样本，让其越小越好
        	gradient_kb(e1_a, e2_a, rel_a, -1);
        	// 梯度更新负样本越大越好
		    gradient_kb(e1_b, e2_b, rel_b, 1);
        }

    //used for relations association by R1 rules
	if (rel2rel.count(rel_a) > 0)
	{
	    for (int i = 0; i < rel2rel[rel_a].size(); i++){
	    // 由规则R1得出的规则与rel_a可以建立规则
		int rel_rpos = rel2rel[rel_a][i].first;
		// the confifidence level of the rule in Rules R1
		// R1规则的置信度
		double rel_pconfi = rel2rel[rel_a][i].second;
		// 计算正样本的E
		double sum_pos = calc_rule(rel_a, rel_rpos);
		// 负采样
		int rel_rneg = rand_max(relation_num);
		// 如果负采样存在对应的规则，继续采样，直到不存在
		while (rule_ok.count(make_pair(rel_a, rel_rneg)) > 0)
			rel_rneg = rand_max(relation_num);
		// 计算负样本的E
		double sum_neg = calc_rule(rel_a, rel_rneg);

		// L3计算
		if (rel_pconfi*sum_pos + marginrule > sum_neg){
			res += margin + rel_pconfi*sum_pos - sum_neg;
			// 进行正负样本梯度更新
			gradient_rule(rel_a, rel_rpos, -lambda_rule);
			gradient_rule(rel_a, rel_rneg, lambda_rule);
		}
		// 单位化
		norm(relation_tmp[rel_a]);
		norm(relation_tmp[rel_rpos]);
		norm(relation_tmp[rel_rneg]);
		// R1 num
		relrules_used++;
	    }
	}
    }

    // 训练路径
    // double x is R(p|h, t)( μi∈B(p) μi)
    void train_path(int rel, int rel_neg, vector<int> rel_path, double margin,double x)
    {
        //正样本的E
        double sum1 = calc_path(rel,rel_path);
        //负样本的E
        double sum2 = calc_path(rel_neg,rel_path);
        //weight of paths and length 2 rules in loss function
	    double lambda = 1;
	    // 计算L2
	    // ？？
        if (sum1+margin>sum2)
        {
        	res+=x*lambda*(margin+sum1-sum2);
        	gradient_path(rel,rel_path, -x*lambda);
		    gradient_path(rel_neg,rel_path, x*lambda);
        }
    }

};

Train train;
void prepare()
{
    FILE* f1 = fopen("./data_FB15K/entity2id.txt","r");
	FILE* f2 = fopen("./data_FB15K/relation2id.txt","r");
	int x;
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{
		string st=buf;
		entity2id[st]=x;
		id2entity[x]=st;
		entity_num++;
	}
	fclose(f1);
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		id2relation[x]=st;
		//加入逆关系,不同的数据集要改这里
		id2relation[x+1345] = "-"+st;
		relation_num++;
	}
	fclose(f2);
	FILE* f_kb = fopen("./data_FB15K/train_pra1.txt","r");
	while (fscanf(f_kb,"%s",buf)==1)
	{
        	string s1=buf;
        	fscanf(f_kb,"%s",buf);
	        string s2=buf;
	        if (entity2id.count(s1)==0)
        	{
            		cout<<"miss entity:"<<s1<<endl;
        	}
	        if (entity2id.count(s2)==0)
        	{
	            	cout<<"miss entity:"<<s2<<endl;
        	}
	        int e1 = entity2id[s1];
        	int e2 = entity2id[s2];
	        int rel;
            fscanf(f_kb,"%d",&rel);
            fscanf(f_kb,"%d",&x);
            vector<pair<vector<int>,double> > b;
            b.clear();
            for (int i = 0; i<x; i++)
            {
			int y,z;
			vector<int> rel_path;
			rel_path.clear();
			fscanf(f_kb,"%d",&y);
			for (int j=0; j<y; j++)
			{
				fscanf(f_kb,"%d",&z);
				rel_path.push_back(z);
			}
			//路径置信度
			double pr;
			fscanf(f_kb,"%lf",&pr);
			// pair<vector<int>,double>
			b.push_back(make_pair(rel_path,pr));
		}
		    //添加三元组及其路径（置信度）
        	train.add(e1,e2,rel,b);
	}
	relation_num*=2;
   
    cout<<"relation_num="<<relation_num<<endl;
    cout<<"entity_num="<<entity_num<<endl;
	
	FILE* f_confidence = fopen("./data_FB15K/confidence.txt","r");
	while (fscanf(f_confidence,"%d",&x)==1)
	{
		string s = "";
		for (int i=0; i<x; i++)
		{
			fscanf(f_confidence,"%s",buf);
			s = s + string(buf)+" ";
		}
		fscanf(f_confidence,"%d",&x);
		for (int i=0; i<x; i++)
		{
			int y;
			double pr;
			fscanf(f_confidence,"%d%lf",&y,&pr);
		    //  s是路径信息 y是关系 pr 是置信度
		//	cout<<s<<' '<<y<<' '<<pr<<endl;
			path_confidence[make_pair(s,y)] = pr;
		}
	}
	fclose(f_confidence);
    fclose(f_kb);

    cout<<"Load all the R1 rules.\n";
    int count_rules = 0;
    FILE* f_rule1 = fopen("./data_FB15K/rule/rule_relation70.txt","r");
	int rel1, rel2, rel3;
	double confi;
        while (fscanf(f_rule1,"%d", &rel1)==1)
        {
                fscanf(f_rule1, "%d%lf", &rel2, &confi);
                rel2rel[rel1].push_back(make_pair(rel2, confi));
                rule_ok[make_pair(rel1, rel2)] = 1;
                count_rules++;
        }
    fclose(f_rule1);

    cout<<"Loading all the R2 rules.\n";
    FILE* f_rule2 = fopen("./data_FB15K/rule/rule_path70.txt","r");
        while (fscanf(f_rule2,"%d%d", &rel1 ,&rel2)==2)
        {
                fscanf(f_rule2, "%d%lf", &rel3, &confi);
                rule2rel[make_pair(rel1, rel2)] = make_pair(rel3, confi);
		        count_rules++;
        }
	cout<<"The confidence of rules is: 0.7"<<"\n";
    cout<<"The total number of rules is: "<<count_rules<<"\n";

    fclose(f_rule2);
}

int main(int argc,char**argv)
{

	cout << "Start to prepare!\n";
        prepare();
	cout << "Prepare Success!\n";
    cout << "Start Training!\n";
        train.run();
	cout << "Training finished.\n";
}
 
