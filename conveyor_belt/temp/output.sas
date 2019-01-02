begin_version
3
end_version
begin_metric
0
end_metric
7
begin_variable
var0
-1
2
Atom atpose(v0, v9)
NegatedAtom atpose(v0, v9)
end_variable
begin_variable
var1
1
2
Atom unsafebtraj(v0, v4, v5, v8, v10)
NegatedAtom unsafebtraj(v0, v4, v5, v8, v10)
end_variable
begin_variable
var2
1
2
Atom unsafebtraj(v1, v6, v7, v13, v15)
NegatedAtom unsafebtraj(v1, v6, v7, v13, v15)
end_variable
begin_variable
var3
-1
2
Atom atpose(v1, v14)
NegatedAtom atpose(v1, v14)
end_variable
begin_variable
var4
-1
3
Atom emptyarm()
Atom picked(v0)
Atom picked(v1)
end_variable
begin_variable
var5
-1
2
Atom inloadingregion(v0)
NegatedAtom inloadingregion(v0)
end_variable
begin_variable
var6
-1
2
Atom inloadingregion(v1)
NegatedAtom inloadingregion(v1)
end_variable
0
begin_state
1
0
0
1
0
1
1
end_state
begin_goal
2
5 0
6 0
end_goal
6
begin_operator
pickup v0 v11 v12
1
5 1
1
0 4 0 1
1
end_operator
begin_operator
pickup v0 v4 v5
1
5 1
1
0 4 0 1
1
end_operator
begin_operator
pickup v1 v16 v17
1
6 1
1
0 4 0 2
1
end_operator
begin_operator
pickup v1 v6 v7
1
6 1
1
0 4 0 2
1
end_operator
begin_operator
place v0 v4 v5 v9 v8 v10
1
1 1
3
0 0 -1 0
0 4 1 0
0 5 -1 0
1
end_operator
begin_operator
place v1 v6 v7 v14 v13 v15
1
2 1
3
0 3 -1 0
0 4 2 0
0 6 -1 0
1
end_operator
2
begin_rule
2
0 1
3 1
1 0 1
end_rule
begin_rule
2
0 1
3 1
2 0 1
end_rule
