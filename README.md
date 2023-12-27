*****************A.1*****************
#include<bits/stdc++.h>
using namespace std;
const int maxn=1e5;
int a[maxn];//原数组 
int b[maxn];//归并排序额外的数组 
int temp[maxn]; 
//快速排序
void quicksort(int a[], int l, int r) {
	if (l >= r) return; //递归出口
	int i = l;
	int j = r;
	int key = a[l];//选择第一个数为key
	while (i < j) {
		while (i < j && a[j] >= key)//从右向左找第一个小于key的值
			j--;
		if (i < j) { //防止左指针超过右指针
			a[i] = a[j];//交换
			i++; //左指针前进
		}
		while (i < j && a[i] < key)//从左向右找第一个大于key的值
			i++;
		if (i < j) { //防止左指针超过右指针
			a[j] = a[i];//交换
			j--;//右指针后退
		}
	}
	a[i] = key; //交换基准数和左右指针重叠的地方，表示左边小于基准数，右边大于基准数，然后进行左右两边的递归过程
	quicksort(a, l, i - 1);//继续排左部分，递归调用
	quicksort(a, i + 1, r);//继续排右部分，递归调用
}
//归并排序-合并
void merge(int low, int mid, int high) //归并
//low 和 mid 分别是要合并的第一个数列的开头和结尾，mid+1 和 high 分别是第二个数列的开头和结尾
{
	int i = low, j = mid + 1, k = low;
	//i、j 分别标记第一和第二个数列的当前位置，k 是标记当前要放到整体的哪一个位置
	while (i <= mid && j <= high)    //如果两个数列的数都没放完，循环
	{
		if (a[i] < a[j])
			b[k++] = a[i++];
		else
			b[k++] = a[j++];   //将a[i] 和 a[j] 中小的那个放入 b[k]，然后将相应的标记变量增加
	}        // b[k++]=a[i++] 和 b[k++]=a[j++] 是先赋值，再增加
	while (i <= mid)
		b[k++] = a[i++];
	while (j <= high)
		b[k++] = a[j++];    //当有一个数列放完了，就将另一个数列剩下的数按顺序放好
	for (int i = low; i <= high; i++)
		a[i] = b[i];                //将 b 数组里的东西放入 a 数组，进行下一轮归并
}
//归并排序递归操作
void mergesort(int x, int y) {
	if (x >= y) return;
	int mid = (x + y) / 2;
	mergesort(x, mid);
	mergesort(mid + 1, y);
	merge(x, mid, y);
}
//希尔排序
void shellsort(int a[], int n){  //a -- 待排序的数组, n -- 数组的长度
	int i, j, gap;   // gap为步长，每次减为原来的一半。
	for (gap = n / 2; gap > 0; gap /= 2){
		// 共gap个组，对每一组都执行直接插入排序
		for (i = 0; i < gap; i++){
			for (j = i + gap; j < n; j += gap){
				// 如果a[j] < a[j-gap]，则寻找a[j]位置，并将后面数据的位置都后移。
				if (a[j] < a[j - gap]) {
					int tmp = a[j];
					int k = j - gap;
					while (k >= 0 && a[k] > tmp){//后移数据找到插入的位置进行插入排序
					
						a[k + gap] = a[k];
						k -= gap;
					}
					a[k + gap] = tmp; //找到了位置进行插入
				}
			}
		}
	}
}
//计数排序
void countsort(int a[],int n){
	vector<int>c; //动态分配内存的数组 
	int maxx=INT_MIN;
	//找出最大值 
	for(int i=0;i<n;i++){
		maxx=max(maxx,a[i]);
	}
	c.resize(maxx+10);//开辟数组 
	for(int i=0;i<n;i++){
		++c[a[i]]; 
	}
	//开始排序
	int k=0; 
	for(int i=0;i<=maxx;i++){ 
		for(int j=0;j<c[i];j++){
			a[k++]=i;
		}
	}
}
//堆排序 
void heapify(int a[], int n, int i) {
    int maxx = i;
    int l = 2 * i + 1, r = 2 * i + 2;
    if (l < n && a[l] > a[maxx]) {
        maxx = l;
    }
    if (r < n && a[r] > a[maxx]) {
        maxx = r;
    }

    if (maxx != i) {
        swap(a[i], a[maxx]);
        heapify(a, n, maxx);
    }
}

void heapsort(int a[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(a, n, i);
    }
    for (int i = n - 1; i >= 0; i--) {
        swap(a[0], a[i]);
        heapify(a, i, 0);
    }
}

void init(int n){
	for(int i=0;i<n;i++){
		a[i]=temp[i];
	}
}
void print(int a[],int n){
	for(int i=0;i<n;i++){
		cout<<a[i]<<" ";
	}
	cout<<"\n";
}

int main() {
	int n;
	cin>>n;
	for(int i=0;i<n;i++){
		cin>>temp[i];
	}
	init(n);
	
	//quicksort(a,0,n-1);
	//shellsort(a,n);
	mergesort(0,n-1);
	//countsort(a,n);
	//heapsort(a,n);
	print(a,n);
	
	
	return 0;
}
**********************A.2**********************
#include <bits/stdc++.h>
using namespace std;
const int maxn=1e6;
vector<string>bucket[maxn]; //拉链法 
vector<string>tb(maxn); //哈希表 
struct info{
	string name;//用户名
	string address;//地址 
	string number; //电话号码-key 
};
/* 线性探测法 */
void conflict1(string &s,int idx){
	int i=idx+1;
	while(i<maxn&&!tb[i].empty()) ++i;
	if(i>=maxn) tb.push_back(s);
	else tb[i]=s;
}
/* 平方探测法 */
void conflict2(string &s,int idx){
	int i=idx,j=1;
	int flag=1;
	while(i>=0&&i<maxn&&!tb[i].empty()){
		i=idx+flag*(j<<1);
		if(flag=-1) ++j;
		flag=-flag;
	}
	if(i>=maxn||i<0) tb.push_back(s);
	else tb[i]=s;
}
/* 拉链法 */
void conflict3(string &s,int idx){
	bucket[idx].push_back(s);
} 
/* ASCII码相加哈希函数 */
int change1(string &s){
	int res=0;
	for(auto &x:s){
		res+=x;
	}
	return res;
}
/* 经典 BKDR字符串哈希 */ 
int change2(string &s) {
    int seed = 31; //质数种子可以减少冲突 
    int res = 0;
    for (int i = 0; i < s.size(); i++) {
       res = res * seed + s[i];
       res%=tb.size(); //防止溢出 
    }
    return res;
}
/* 插入 */
void insert(string &s){
	cout<<"输入你要选择的哈希函数"<<"\n";
	cout<<"输入1:ASCII码相加哈希函数，输入2：经典 BKDR字符串哈希"<<"\n"; 
	int y,idx;
	cin>>y;
	if(y==1) idx=change1(s);
	else idx=change2(s);
	if(!tb[idx].empty()){
		cout<<"产生矛盾，请选择你的解决方法"<<"\n";
		cout<<"输入1:线性探测法，输入2：平方探测法，输入3：拉链法"<<"\n";
		int x;
		cin>>x;
		if(x==1) conflict1(s,idx); //线性探测法 
		else if(x==2) conflict2(s,idx); //平方探测法 
		else conflict3(s,idx); //拉链法 
	}
	tb[idx]=s;
	bucket[idx].push_back(s); //拉链法插入 
}
int main() {
    string s;
    cout << "请输入要插入的字符串：" << endl;
    cin >> s;

    cout << "请选择哈希函数类型：" << endl;
    cout << "输入 1：ASCII码相加哈希函数" << endl;
    cout << "输入 2：经典 BKDR字符串哈希" << endl;
    int y;
    cin >> y;

    if (y != 1 && y != 2) {
        cout << "无效的哈希函数选择！" << endl;
        return 0;
    }
    int idx;
    if (y == 1) {
        idx = change1(s);
    } else {
        idx = change2(s);
    }
    insert(s);

    // 输出哈希表中的数据
    for (int i = 0; i < tb.size(); i++) {
        if (!tb[i].empty()) {
            cout << "哈希表[" << i << "]：" << tb[i] << endl;
        }
    }

    // 输出每个哈希桶中的数据
    for (int i = 0; i < maxn; i++) {
        if (!bucket[i].empty()) {
            cout << "哈希桶[" << i << "]：";
            for (int j = 0; j < bucket[i].size(); j++) {
                cout << bucket[i][j] << " ";
            }
            cout << endl;
        }
    }

    return 0;
}
*******************B.1*******************
#include<bits/stdc++.h>
using namespace std;
const int maxn = 1.2e7 +10;
const int maxm = 5e5 + 10;
//#define int long long
#define MIN(a,b)  (((a)<(b))?(a):(b))
#define MAX(a,b)  (((a)>(b))?(a):(b))
/* Manacher算法 */
string str; //原串
char s[maxn<<1]; //通过Manacher转化的字符串,扩大2倍
int p[maxn<<1]; //回文串半径，扩大2倍
int C; //中心
int R; //最右端
int n;
void change() { //转换字符串为Manacher模式字符串
	int k = 0;
	s[k++] = '$';//插入无用字符，作为开头结束标志
		s[k++]='#';
	for (auto& ch : str) {
		s[k++] = ch;
		s[k++] = '#';
	}
	s[k++] = '&';//结尾插入无用字符，作为结束标志
	n = k;
}
void Manacher() { //Manacher算法
	for (int i = 1; i < n; i++) {
		if (i < R) { //如果i在扩展回文串中，则可以利用对称性寻找回文串
			p[i] = MIN(p[(C << 1) - i], p[C]+C-i);
		}
		else p[i] = 1; //如果不在扩展中，赋值1
		while (s[i + p[i]] == s[i - p[i]]) { //中心扩展
			p[i]++; 
		}
		if (p[i] + i > R) { //如果需要更新最右端
			R = p[i] + i;//更新扩展的最右端
			C = i;//更新中心点
		}
	}
	int len = 1,maxIndex=-1;
	for (int i = 0; i < n; i++) {
		if(p[i]-1>len) {
			len =p[i]-1; //这里得到的回文串会比原来大1
			maxIndex=i;
		}
	}
	int mid=(maxIndex-1)/2;
	int left,right;	
	if(len%2==0) left=mid-1,right=mid;
	else left=mid-1,right=mid+1;
	while(left>=0&&right<n&&str[left]==str[right]){
		left--;
		right++;
	}
	for(int i=left+1;i<right;i++){
		cout<<str[i];
	}
	cout<<"\n";
}
bool check(string s){
	int l=0,r=s.size()-1;
	while(l<=r&&s[l]==s[r]){
		l++,r--;
	}	
	return l>=r;
}
void force(){
	int n=str.size();
	string res;
	for(int i=0;i<n;i++){
		string temp;
		temp+=str[i];
		for(int j=i+1;j<n;j++){
			temp+=str[j];
			if(check(temp)){
				if(temp.size()>res.size()){
					res=temp;
				}	
			}
		}
	}
	cout<<res<<"\n";
}
signed main() {
	//ios::sync_with_stdio(0);
	cin.tie(0);
	cout.tie(0);
	cout<<"输入字符串"<<"\n"; 
	cin >> str;
	cout<<"暴力匹配"<<"\n";
	force();//暴力匹配 
	change(); //转换Manacher回文串
	cout<<"Manacher"<<"\n";
	Manacher(); //进行Manacher算法
}
***************************B.2***************************
#include<bits/stdc++.h>
using namespace std;
const int maxn = 1e5+10;
const int maxm = 2e5+10;
int n, m;//点和边
int s; //初始点
int dis[maxn];//起点距离每个点的距离
const int INF=99999; //无穷大
struct edge {
    int u, v, w; // 边的起点、终点和权值
}e[maxm];
// Bellman-Ford单源最短路径算法 
void bellmanFord() {
    // 初始化dis数组
    for(int i=1;i<=n;i++){
    	dis[i]=INF;
	}
	dis[s] = 0; // 初始点到自身的距离为0
    for (int i = 1; i <= n - 1; i++) { // 进行n-1轮松弛操作
        for (int j = 1; j <= m; j++) {
            if (dis[e[j].u] + e[j].w < dis[e[j].v]) { //更新最短路径 
                dis[e[j].v] = dis[e[j].u] + e[j].w;
            }
        }
    }
}
int main() {
	cin >> n >> m>>s;
	/* 有向图建图 */
	for (int i = 1; i <= m; i++) { //n个点，m条边
		cin>>e[i].u>>e[i].v>>e[i].w;
	}
	
	bellmanFord();//跑一遍Bellman-Ford可以得到s到所有点的最短路径
	/* 打印最短路径 */
	dis[s]=0;
	for (int i = 1; i <= n; i++) {
		if(dis[i]==INF) cout<<"INF"<<"\n";
		else cout << dis[i] << "\n";
	}
	cout << "\n";
	
	return 0;
}
*******************B.3*************************
#include<bits/stdc++.h>
using namespace std;
const int maxn=1e3+10;
const int maxm=1e3+10;
int n,m;
int x1,x2,y1,y2;
int res=INT_MAX;
int a[maxn][maxm];
int vis[maxn][maxm];

struct ss{
	int x,y,cnt;
};

bool check(int i,int j){
	if(i>n||i<=0||j>m||j<=0) return false;
	if(a[i][j]==1||vis[i][j]) return false;
	return true;
}
void dfs(int x, int y, int cnt) {
    if (!check(x, y) || cnt >= res) return;
    vis[x][y] = 1;
    if (x == x2 && y == y2) {
        res = min(res, cnt);
        return;
    }
    if (check(x+1, y)) {
        dfs(x+1, y, cnt+1);
        vis[x+1][y] = 0;
    }
    if (check(x-1, y)) {
        dfs(x-1, y, cnt+1);
        vis[x-1][y] = 0;
    }
    if (check(x, y+1)) {
        dfs(x, y+1, cnt+1);
        vis[x][y+1] = 0;
    }
    if (check(x, y-1)) {
        dfs(x, y-1, cnt+1);
        vis[x][y-1] = 0;
    }
}
void bfs(){
	queue<ss>q;
	if(check(x1,y1)) q.push({x1,y1,0});
	while(!q.empty()){
		ss node=q.front();q.pop();
		if(node.x==x2&&node.y==y2){
			res=min(res,node.cnt);
		}
		if(check(node.x+1,node.y)){
			q.push({node.x+1,node.y,node.cnt+1});
			vis[node.x+1][node.y]=1;
		}
		if(check(node.x-1,node.y)){
			q.push({node.x-1,node.y,node.cnt+1});
			vis[node.x-1][node.y]=1;	
		} 
		if(check(node.x,node.y+1)){
			q.push({node.x,node.y+1,node.cnt+1});
			vis[node.x][node.y+1]=1;
		}
		if(check(node.x,node.y-1)) {
			q.push({node.x,node.y-1,node.cnt+1});
			vis[node.x][node.y-1]=1;
		}
	}
}
int main() {
	cin>>n>>m;
	for(int i=1;i<=n;i++){
		for(int j=1;j<=m;j++){
			cin>>a[i][j];
		}
	}
	cout<<"输入起点"<<"\n"; 
	cin>>x1>>y1;
	cout<<"输入终点"<<"\n";
	cin>>x2>>y2; 
	cout<<"dfs路径长度"<<"\n";
	dfs(x1,y1,0);
	if(res==INT_MAX){
		cout<<"无法到达"<<"\n";
	}
	else cout<<res<<"\n";
	res=INT_MAX;
	memset(vis,0,sizeof(vis));
	cout<<"bfs路径长度"<<"\n";
	bfs();
	if(res==INT_MAX){
		cout<<"无法到达"<<"\n";
	}
	else cout<<res<<"\n";
	return 0;
}
******************B.4******************
#include<bits/stdc++.h>
using namespace std;
vector<vector<string>> res;
bool check(const vector<string>& track, int row, int col, int n) {
    for (int i = 0; i < row; i++) {
        if (track[i][col] == 'Q') {
            return false;
        }
    }
    for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
        if (track[i][j] == 'Q') {
            return false;
        }
    }
    for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
        if (track[i][j] == 'Q') {
            return false;
        }
    }
    return true;
}

void backtrack(int row, vector<string>& track, int n) {
    if (row == n) {
        res.push_back(track);
        return;
    }
    for (int col = 0; col < n; col++) {
        if (!check(track, row, col, n)) {
            continue;
        }
        track[row][col] = 'Q';
        backtrack(row + 1, track, n);
        track[row][col] = '.';
    }
}

vector<vector<string>> solveNQueens(int n) {
    vector<string> track(n, string(n, '.'));
    backtrack(0, track, n);
    return res;
}

int main() {
    vector<vector<string>> s = solveNQueens(8);
    cout << s.size() << "\n";
    for (int i = 0; i < s.size(); i++) {
        cout << i + 1 << " : ";
        for (int j = 0; j < s[i].size(); j++) {
            for (int k = 0; k < s[i][j].size(); k++) {
                if (s[i][j][k] == 'Q') {
                    cout << "(" << j << "," << k << ")" << " ";
                }
            }
        }
        cout << "\n";
    }
    return 0;
}
******************B.5******************
#include<bits/stdc++.h>
using namespace std;
const int maxn=1e5;
struct node{
	int left,right,val;
}tree[maxn];
//先序 
void dfs1(int i){
	if(i==-1) return;
	cout<<tree[i].val<<" ";
	dfs1(tree[i].left);
	dfs1(tree[i].right);
}
//中序
void dfs2(int i){
	if(i==-1) return;
	dfs2(tree[i].left);
	cout<<tree[i].val<<" ";
	dfs2(tree[i].right);
}
//后序 
void dfs3(int i){
	if(i==-1) return;
	dfs3(tree[i].left);
	dfs3(tree[i].right);
	cout<<tree[i].val<<" ";
}
int main(){
	int n;
	cin>>n;
	for(int i=1;i<=n;i++){
		cin>>tree[i].val>>tree[i].left>>tree[i].right;
	}
	cout<<"前序遍历"<<"\n";
	dfs1(1);
	cout<<"\n";
	cout<<"中序遍历"<<"\n";	
	dfs2(1);
	cout<<"\n";
	cout<<"后序遍历"<<"\n";
	dfs3(1);
	return 0;
}
*************B.7************
#include<bits/stdc++.h>
using namespace std;
const int maxn = 1e5;
int a[maxn];
int main() {
	int n;
	cin >> n;
	for (int i = 1; i <= n; i++) {
		cin >> a[i];
	}
	bool flag = true;
	int cnt = 0;
	for (int i = 1; i < n; i++) {
		int minIndex = i, minn = a[i];
		for (int j = i; j <= n; j++) {
			if (a[j] < minn) {
				minn = a[j];
				minIndex = j;
			}
		}
		for (int k = minIndex; k > i; k--) {
			swap(a[k], a[k - 1]);
			cnt++;
		}
		cout<<"第"<<i<<"遍:";
		for(int i=1;i<=n;++i) cout<<a[i]<<" ";
		cout<<"\n";
	}
	cout << "共移动"<<cnt << "步\n";
	return 0;
}
********************B.8*******************
#include<bits/stdc++.h>
using namespace std;
const int maxn = 1e5 + 10;
const int maxm = 1e5 + 10;
//#define int long long
#define MIN(a,b)  (((a)<(b))?(a):(b))
#define MAX(a,b)  (((a)>(b))?(a):(b))
/* Kosaraju算法求强连通分量 */
vector<int>e[maxn];//原图
vector<int>re[maxn];//逆图
vector<int>S;  //存储第一次DFS：标记点的顺序
bitset<maxn>vis;
int mark[maxn];//标记强连通分量
vector<int>scc[maxn]; //强连通分量元素
int cnt; //强连通分量的个数
void dfs1(int u) { //用于得到初始的顺序
	if (vis[u]) return;
	vis[u] = 1;
	for (auto& v : e[u]) {
		dfs1(v);
		S.push_back(u); //记录点的递归访问顺序
	}
}
void dfs2(int u) {//用于标记每一块强连通分量
	if (mark[u]) return; //强连通分量已经标记了
	mark[u] = cnt; //开始标记
	scc[cnt].push_back(u);
	for (auto& v : re[u]) { //遍历逆图
		dfs2(v);
	}
}
void Kosaraju(int n) { //Kosaraju算法
	for (int i = 1; i <= n; i++) {
		dfs1(i); //遍历原图得到顺序
	}
	for (int i = n - 1; i >= 0; i--) {//注意这里是倒着遍历
		if (!mark[S[i]]) { //寻找隔离块
			cnt++;
			dfs2(S[i]); //进行标记
		}
	}
}
signed main() {
	int n, m;
	cin >> n >> m;
	while (m--) {
		int x, y;
		cin >> x >> y;
		e[x].push_back(y); //建立原图
		re[y].push_back(x);//建立逆图
	} 
	Kosaraju(n);//开始跑Kosaraju算法
	if(cnt==1) cout<<"Yes"<<"\n";
	else cout<<"No"<<"\n"; 
}
****************B.13*****************
#include <bits/stdc++.h>
using namespace std;
const int maxn = 1e5 + 10;
const int maxm = 2e5 + 10;
int n, m;  //点和边
int s;      //初始点
int dis[maxn]; //起点距离每个点的距离
int vis[maxn];
int cnt[maxn]; //记录每个点入队次数
#define INF -1 //无穷小 
struct edge {
    int v, dis; //终点，边权
};
vector<edge> e[maxn];
/* SPFA单源最长路径算法 */
bool spfa() {
    queue<int> q;  //利用到STL队列
    for (int i = 1; i <= n; i++) {
        dis[i] = INF; //初始化无穷小
        vis[i] = 0;   //记录点是否在队列中
        cnt[i] = 0;   //记录每个点入队次数
    }
    q.push(s); 
    vis[s] = 1; 
    dis[s] = 0; //初始化
    while (!q.empty()) {
        int u = q.front(); 
        q.pop(); 
        vis[u] = 0; //取出队头
        for (auto &x : e[u]) { //遍历图
            int v = x.v;
            if (dis[u] + x.dis > dis[v]) { //更新距离
                dis[v] = dis[u] + x.dis;
                if (vis[v] == 0) { //重新入队
                    vis[v] = 1;
                    q.push(v);
                    cnt[v]++;  // 入队次数加1
                    if (cnt[v] >= n) { // 如果入队次数超过n，说明存在正环
                        return true;
                    }
                }
            }
        }
    }
    return false;
}
int main() {
	int d;
    cin >> n >> m;
	cout<<"输入起点"<<"\n";
	cin>>s;
	cout<<"输入终点"<<"\n"; 
	cin>>d;
    /* 有向图建图 */
    for (int i = 1; i <= m; i++) { //n个点，m条边
        int u, v, w;
        cin >> u >> v >> w;
        e[u].push_back({v, w});
    }
    if (spfa()) {
        cout << "存在正环" << endl;
    } 
	else{
        /* 打印最长路径 */
    	cout<<dis[d]<<"\n";
    }
    return 0;
}
*****************C.14********************
#include<bits/stdc++.h>
using namespace std; 
int cnt[4]; //25 10 5 1
int main() {
	int x;
	cout<<"请输入钱数"<<"\n";
	cin>>x;
	cnt[0]+=x/25; x%=25;
	cnt[1]+=x/10; x%=10;
	cnt[2]+=x/5; x%=5;
	cnt[3]+=x;
	cout<<"25元、10元、5元、1元的零钱如下"<<"\n";
	for(int i=0;i<4;i++){
		cout<<cnt[i]<<" ";
	}
	cout<<"\n";
	
	return 0;
}
****************C.18*****************
#include<bits/stdc++.h>
using namespace std;
int main() {
	cout<<"共有10个人:";
	for(int i=1;i<=10;i++) cout<<i<<" ";
	cout<<"\n";
	cout<<"每隔3个人淘汰一个人"<<"\n";
	list<int>a;
	for(int i=1;i<=10;i++) a.push_back(i);
  	auto pos = a.begin();
  	cout<<"淘汰顺序: ";
	while (a.size()>1) {
		for (int i = 1; i < 3; i++) {
	        pos++;
	        if (pos == a.end()) {
	            pos = a.begin();
	        }
	    }
	    auto next_pos = pos;
	    if (++next_pos == a.end()) {
	        next_pos = a.begin();
	    }
	    cout << *pos <<" ";
	    pos = a.erase(pos);
	    if (pos == a.end()) {
	        pos = a.begin();
	    }
	}
    cout<<"\n";
    cout<<"最后剩下的人: "<<*pos<<"\n";
    return 0;
}

