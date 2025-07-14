# 入力 x x x 横一列
# int型のリストになる [1, 6  10]
list(map(int, input().split()))

# 入力 縦一列
# x x
# x x
# x x
# [[1, 6], [4, 5], [5, 10], [7.10]]
[map(int, input().split()) for _ in range()]

# 普通の入力
int(input())

float(input())

bool(input())


# 文字列入力 378 → a, b, c = input()で受け取れる
input()

# 空白区切りで配列 3 7 8→ [3, 7, 8]
input().split()

# アンパック代入も可能
x,y = input().split() 


