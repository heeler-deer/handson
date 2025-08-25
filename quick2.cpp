#include <vector>
#include <stack>
#include <utility> // for pair
#include <iostream>


//非递归快排



using namespace std;

// 分区函数
int partition(int arr[], int left, int right) {
    int pivot = arr[(left + right) / 2]; // 选择中间元素作为基准
    swap(arr[(left + right) / 2], arr[right]); // 将基准移到末尾
    
    int store_index = left;
    for (int i = left; i < right; i++) {
        if (arr[i] < pivot) {
            swap(arr[i], arr[store_index]);
            store_index++;
        }
    }
    
    swap(arr[store_index], arr[right]); // 将基准放回正确位置
    return store_index;
}

// 非递归快速排序主函数
void quick_sort(int arr[], int size) {
    if (size <= 1) return;
    
    stack<pair<int, int>> stk;
    stk.push(make_pair(0, size - 1));
    
    while (!stk.empty()) {
        auto [left, right] = stk.top();
        stk.pop();
        
        if (left >= right) continue;
        
        int pivot_index = partition(arr, left, right);
        
        // 先处理较大的子区间以减少栈深度
        if (pivot_index - left > right - pivot_index) {
            stk.push(make_pair(left, pivot_index - 1));
            stk.push(make_pair(pivot_index + 1, right));
        } else {
            stk.push(make_pair(pivot_index + 1, right));
            stk.push(make_pair(left, pivot_index - 1));
        }
    }
}




int main() {
    int data[] = {9, 3, 7, 4, 5, 8, 2, 1, 6};
    int size = sizeof(data) / sizeof(data[0]);
    
    cout << "Before sorting: ";
    for (int i = 0; i < size; i++) {
        cout << data[i] << " ";
    }
    cout << endl;
    
    quick_sort(data, size);
    
    cout << "After sorting: ";
    for (int i = 0; i < size; i++) {
        cout << data[i] << " ";
    }
    cout << endl;
    
    return 0;
}