class NumArray {
private:
    vector<int> nums;
    vector<int> tree;
    int prefixSum(int i){
        int s=0;
        for(;i>0;i&=i-1){
            s+=tree[i];
        }
        return s;
    }


public:
    NumArray(vector<int>& nums) :nums(nums),tree(nums.size()+1){
        for(int i=1;i<=nums.size();i++){
            tree[i]+=nums[i-1];
            int nxt=i+(i&-i);
            if(nxt<=nums.size()){
                tree[nxt]+=tree[i];
            }
        }
    }
    
    void update(int index, int val) {
        int delta=val-nums[index];
        nums[index]=val;
        for(int i=index+1;i<tree.size();i+=i&-i){
            tree[i]+=delta;
        }
    }
    
    int sumRange(int left, int right) {
        return prefixSum(right+1)-prefixSum(left);
    }
};

// 每个下标 i 维护的值是 [i - (i & -i) + 1, i] 这个区间的前缀和。

// i & -i 提取的是 i 的最低位的1所代表的值，比如：

// i = 6 (110)，i & -i = 2

// i = 8 (1000)，i & -i = 8




// i &= (i - 1) 会消除最低位的 1
// 这是在不断往上找“包含当前点的前缀区间”。

// 例子：

// i = 6 (110)，你加上 tree[6]，然后 i &= i - 1 → i = 4

// tree[6] 是 [5,6] 的和，tree[4] 是 [1,4] 的和

// 所以相加就得到了 [1,6] 的前缀和