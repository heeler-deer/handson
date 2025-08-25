    //10 to other<=36
    string base_repr(int v, int base) {
        string s;
        while (v > 0) {
            int d = v % base;
            s += d < 10 ? '0' + d : 'A' + d - 10;
            v /= base;
        }
        reverse(s.begin(),s.end());
        return s;
    }