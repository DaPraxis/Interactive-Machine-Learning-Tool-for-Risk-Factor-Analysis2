def climbStairs(self, n: int) -> int:
        climb_dic = {}
       
        for i in range(n+1):
            if i <= 2:
                climb_dic[i] = i
            if i == 3:
                climb_dic[i] = 4
            if i not in climb_dic:
                climb_dic[i] = climb_dic[i-1] + climb_dic[i-2] + climb_dic[i-3]
        
        return climb_dic[n]