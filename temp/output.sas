begin_version
3
end_version
begin_metric
1
end_metric
11
begin_variable
var0
-1
2
Atom picked(v0, #o12, #o13)
NegatedAtom picked(v0, #o12, #o13)
end_variable
begin_variable
var1
-1
2
Atom picked(v1, #o5, #o6)
NegatedAtom picked(v1, #o5, #o6)
end_variable
begin_variable
var2
-1
2
Atom picked(v2, #o7, #o8)
NegatedAtom picked(v2, #o7, #o8)
end_variable
begin_variable
var3
-1
2
Atom picked(v3, #o22, #o23)
NegatedAtom picked(v3, #o22, #o23)
end_variable
begin_variable
var4
-1
2
Atom picked(v4, #o20, #o21)
NegatedAtom picked(v4, #o20, #o21)
end_variable
begin_variable
var5
-1
2
Atom emptyarm()
NegatedAtom emptyarm()
end_variable
begin_variable
var6
-1
2
Atom inloadingregion(v0)
NegatedAtom inloadingregion(v0)
end_variable
begin_variable
var7
-1
2
Atom inloadingregion(v1)
NegatedAtom inloadingregion(v1)
end_variable
begin_variable
var8
-1
2
Atom inloadingregion(v2)
NegatedAtom inloadingregion(v2)
end_variable
begin_variable
var9
-1
2
Atom inloadingregion(v3)
NegatedAtom inloadingregion(v3)
end_variable
begin_variable
var10
-1
2
Atom inloadingregion(v4)
NegatedAtom inloadingregion(v4)
end_variable
0
begin_state
1
1
1
1
1
0
1
1
1
1
1
end_state
begin_goal
5
6 0
7 0
8 0
9 0
10 0
end_goal
10
begin_operator
pickup_obj_four v4 #o20 #o21 v0 v1 v2 v3
5
6 0
7 0
8 0
9 0
10 1
2
0 5 0 1
0 4 -1 0
1
end_operator
begin_operator
pickup_obj_one v1 #o5 #o6 v0 v2 v3 v4
5
6 0
7 1
8 1
9 1
10 1
2
0 5 0 1
0 1 -1 0
1
end_operator
begin_operator
pickup_obj_three v3 #o22 #o23 v0 v1 v2 v4
5
6 0
7 0
8 0
9 1
10 1
2
0 5 0 1
0 3 -1 0
1
end_operator
begin_operator
pickup_obj_two v2 #o7 #o8 v0 v1 v3 v4
5
6 0
7 0
8 1
9 1
10 1
2
0 5 0 1
0 2 -1 0
1
end_operator
begin_operator
pickup_obj_zero v0 #o12 #o13 v1 v2 v3 v4
4
7 1
8 1
9 1
10 1
2
0 5 0 1
0 0 -1 0
1
end_operator
begin_operator
place v0 #o12 #o13 #o15 #o14 #o16
0
3
0 5 -1 0
0 6 -1 0
0 0 0 1
1
end_operator
begin_operator
place v1 #o5 #o6 #o18 #o17 #o19
0
3
0 5 -1 0
0 7 -1 0
0 1 0 1
1
end_operator
begin_operator
place v2 #o7 #o8 #o10 #o9 #o11
0
3
0 5 -1 0
0 8 -1 0
0 2 0 1
1
end_operator
begin_operator
place v3 #o22 #o23 #o25 #o24 #o26
0
3
0 5 -1 0
0 9 -1 0
0 3 0 1
1
end_operator
begin_operator
place v4 #o20 #o21 #o28 #o27 #o29
0
3
0 5 -1 0
0 10 -1 0
0 4 0 1
1
end_operator
0
