from math import ceil 
import re
'''
1. 問題A_086：Product（100点）・・・解けた
時間計算量：O(1)

二つの正整数が与えられます。 aとbの 
の積が偶数か奇数か判定する
'''

def product(a, b) -> str:
  return 'Even' if (a * b) % 2 == 0 else 'Odd'


'''
2. 問題A_081：Placing Marbles（100点）・・・解けた
時間計算量：O(1)

0と1のみからなる3桁の番号sが与えられる。
1が何個含まれるか求めよ。
'''
def placing_marbles(s: str) -> int:
  return int(s[0]) + int(s[1]) + int(s[2])

'''
問題A_042：和風いろはちゃんイージー・・・解けた
時間計算量：O(1)
入力された正数が575に並べ替えられるか

575の判定を5+7+5=17になれば575と言える

'''

def iroha_a(a, b, c) -> str:
  return 'YES' if a + b + c == 17 else 'NO'

'''
問題A_043：キャンディーとN人の子供イージー
時間計算量：O(1)

等差数列で解ける
'''

def n_candy(N: int) -> int:
  return N * (1 + N) // 2


'''
問題A_044：高橋君とホテルイージー 
時間計算量：O(1)
'''

def hotel_easy(N: int, K: int, X: int, Y: int) -> int:
  return X * K + (N - K) * X if N < K else X * K + (N - K) * Y

N,K,X,Y = [int(input()) for _ in range(4)]
print((hotel_easy(N,K,X,Y)))

'''
問題A_045：台形の面積
'''

def daikei_area(a: int, b: int, h: int) -> int:
  return h * (a + b) // 2

a,b,h = [int(input()) for _ in range(3)]
print(daikei_area(a, b, h))

'''
問題A _046：AtCoDeerくんとペンキ

データ構造をsetを利用することで重複データを排除
'''
color = set()
a,b,c = map(int, input().split())
color.add(a)
color.add(b)
color.add(c)
print(len(color))

'''
問題A_047：キャンディーと2人の子供
'''
candy = input().split()
candy.sort(reverse=True)
q = int(candy[0]) == int(candy[1]) + int(candy[2])
print({'Yes' if q else 'No'})

'''
問題A_048：AtCoder *** Contest
略称の出力
'''

a,b,c = input().split()
print(a[0] + b[0] + c[0])

'''
問題A_049：居合を終え、青い絵を覆う
'''
s = input()
print('vowel' if s in ['a', 'e', 'i', 'o', 'u'] else 'consonant')

'''
問題A_050：Addition and Subtraction Easy
'''

a,p,b = input().split()
n_a, n_b = int(a), int(b)
print(n_a + n_b if p == '+' else n_a - n_b)

'''
問題A_051：Haiku
文字列操作（範囲指定）
'''
s = input().split(',')
s1,s2,s3 = s[0][:5],s[1][:7],s[2][:]
print(s1 + ' ' +  s2 + ' ' + s3)

'''
問題A_052：Two Rectangles
長方形の面積
'''

def s_rectangle(A: int, B: int, C: int, D:int) -> int:
  s_ab, s_cd  = A * B, C * D
  return s_cd if s_ab <= s_cd else s_ab

A,B,C,D = map(int, input().split())
print(s_rectangle(A, B, C, D))

'''
問題A_053：ABC/ARC
比較演算子 以上・以下・未満に含まれるかの問題
'''

X - int(input())
print('ABC' if X <= 1199 else 'ARC')

'''
問題A_054：One Card Poker
弱 2 < 3 < 4 < 5 < 6 < 7 < 8 < 9 < 10 < 11 < 12 < 13 < 1 強

パターン：
1あり・・・・・・1を特定し格納
1なし・・・・・・大小の比較
数字が同じ・・・・等しいか
'''
A, B = map(int, input().split())

def one_card_poker(player: list) -> str:
  A, B = int(player[0]), int(player[1])
  if A == B:
    return 'Draw'
  if '1' not in player:
    return 'Alice' if A > B else 'Bob'
  
  card_1 = player[player.index('1')]
  
  return 'Alice' if str(A) == card_1 else 'Bob'
  
player = input().split()
print(one_card_poker(player))

'''
問題A_055：Restaurant
(N*800)-(N//15)*200
'''

N = int(input())
print((N*800) - (N//15)*200)

'''
問題A_056：
条件のパターン問題
HH = H
HD = D
DH = D
DD = H
'''

def honesty_liar(player: list) -> str:
  A, B = player[0], player[1]
  if A == 'H':
    return B
  elif A == 'D' and B == 'D':
    return 'H'
  else:
    return 'D'

player = input().split()
print(honesty_liar(player))

'''
問題A_057：Remaining Time
24時間表記で時刻を表示する
'''

a,b = map(int, input().split())
print((a + B) % 24)

'''
問題A_058：ι⊥l

b - a = c - b
'''

a,b,c = map(int, input().split())
print('YES' if b - a == c - b else 'NO')

'''
問題A_059：Three-letter acronym
文字列変換＋連結
'''

s1, s2, s3 = input().split()
print(s1[0].upper() + s2[0].upper() + s3[0].upper())

'''
問題A _060：Shiritori
'''
a,b,c = input().split()
print('YES' if a[-1] == b[0] and b[-1] == c[0] else 'NO')


'''
問題A_061：Between Two Integers
'''

a,b,c = map(int, input().split())
print('Yes' if a <= c <= b else 'No')

'''
問題A_062：A - Grouping
'''
x,y = map(int, input().split())
s = ''
a = [1, 3, 5, 7, 8, 10, 12]
b = [4, 6, 9, 11]
if x in a and y in a:
  s = 'Yes'
elif x in b and y in b:
  s = 'Yes'
else:
  s = 'No'
print(s)

'''
問題A_063：Restricted
'''
a, b = map(int, input().split())
c = a + b
print(c if c < 10 else 'error')

'''
問題A_064：RGB Cards
'''

r,g,b = map(int, input().split())
calc = int(str(r) + str(g) + str(b))
print(calc)

'''
問題A_065：Expired?
'''

x,a,b = map(int, input().split())
calc = b - a
result = ''
if calc <= 0:
  result = 'delicious'
elif calc <= x:
  result = 'safe'
elif calc >= x + 1:
  result = 'dangerous'
print(result)

'''
問題A_066：ringring
a - b - c
b - c
c ×
(a, b),(a, c)
(b, c)
'''
a,b,c = map(int, input().split())
min_1, min_2, min_3 = a + b, a + C, b + C
q = [min_1, min_2, min_3]
print(min(q))

'''
問題A_067：Sharing Cookies
A, B, A + Bの判定が必要
'''
a,b = map(int, input().split())
if a % 3 == 0 or b % 3 == 0 or (a + b) % 3 == 0:
  print('Possible')
else:
  print('Impossible')

'''
問題A_068：ABCxxx
'''
s = input()
print('ABC' + s)

'''
問題A_069：K-City＊＊＊
n：たて m：横
横方向 = n - 1行
縦方向 = m - 1列
(n - 1) * (m - 1)
'''

n,m = map(int, input().split())
print((n - 1) * (m - 1))

'''
問題A_070：Palindromic Number
回文：逆から読んでも同じ
3つの整数なら先頭と末尾が同じなら真ん中が必然的に同じとなり
回文であることが証明できる
回文しても真ん中の位置は変わらない＝だから先頭と末尾だけこの場合考えればいい
'''
n = input()
print('Yes' if n[0] == n[-1] else 'No')

'''
問題A_071：Meal Delivery
絶対値の考え方
'''
x,a,b = map(int, input().split())
c = abs(x - a); d = abs(x - b)
print('A' if c <= d else 'B')

'''
問題A_072：Sandglass2
'''
x,t = map(int, input().split())
print(x - t if x >= t else 0)

'''
問題A_073：September 9？？
シンプルに考える
2桁の先頭と末尾が9だったらで分かる
'''
n = input()
print('Yes' if n[0] == '9' or n[1] == '9' else 'No')

'''
問題A_074：Bichrome Cells
N^2 - A
(N * N) - A
'''

N,A = [int(input()) for _ in range(2)]
print((N**2)-A)

'''
問題A_075：One out of Three（苦手）＊＊
3つの整数から1つだけ異なる整数を見つける
1つの同じペアが確定すれば残りが違う整数と判断できる
→残りを考える！！

パターンの問題：
A == B：C
A == C：B
それ以外はA
'''

A,B,C = map(int, input().split())
N = 0
if A == B:
  N = C
elif A == C:
  N = B
else:
  N = A
print(N)

'''
問題A_076：Rating Goal
(a + b) // 2 = PF
bを求める為に両辺にかける

b = PF*2 - a
PF + b = 次のコンテストのパフォーマンス
'''

R,G = [int(input()) for _ in range(2)]
print((G * 2) - R)

'''
問題A_077：Rotation

回文
'''

c1, c2 = [input() for _ in range(2)]
print('YES' if c1 == c2[::-1] else 'NO')

'''
問題A_078：HEX（16進数）

データ構造：辞書
'''

hex = {'A' : 10, 'B' : 11, 'C' : 12, 'D' : 13, 'E': 14, 'F': 15}
x,y = input().split()
s = ''
if hex[x] < hex[y]:
  s = '<'
elif hex[x] > hex[y]:
  s = '>'
else:
  s = '='
print(s)

'''
問題A_079：Good Integer
1118
2998
'''

str_n = input()
a ='Yes' if str_n[0] == str_n[1] == str_n[2] or str_n[1] == str_n[2] == str_n[3] else 'No'
print(a)

'''
問題A_080：Parking
'''

N,A,B = map(int, input().split())
p1 = N * A
print(p1 if p1 <= B else B)

'''
問題A_407：
四捨五入の問題
'''
a,b = map(int, input().split())
print(round(a / b ))

'''
問題A_406：Not Acceptable（47分)

if文を書く前にパターンを想定される洗い出す
方針：
時を見る
提出時(c)の方が締切時(a)より小さいならその時点でYesと確定する
提出時と締切時が同じだったら分を判定する

（解法）
データ構造：tuple
先頭から要素を判定する

もしくは分へ変換
例）
22*60+40 < 22*60+30 
'''

a,b,c,d = map(int, input().split())
s = ''
print('Yes' if (a ,b) < (c, d) else 'No')

'''
問題A_405：Is it rated?

ARC Div1:1600 <= 2999
    Div2:1200 <= 2399

Rated対象かどうか
方針：
Div1の場合とDiv2の場合を分ける
 
それ以下はNo
'''
r,x = map(int, input().split())
s = ''
if x == 1 and 1600 <= r <= 2999:
  s = 'Yes'
elif x == 2 and 1200 <= r <= 2399:
  s = 'Yes'
else:
  s = 'No'
print(s)

'''
問題A_405：Not Found
時間計算量：O(1)
'''
s = set(input())
c = set(chr(ord("a")+i) for i in range(26))
ans = list(c - s)
print(ans[0])

'''
問題A_403：Odd Position Sum 
奇数番目の総和

偶数番目の要素を足し合わせる
'''

n = int(input())
a = map(int, input().split())
print(sum (aa for i, aa in enumerate(a) if i % 2 == 0))

'''
問題A_402：

方針：
正規表現で大文字の抽出
文字列の空白を削除し出力
'''

s = input()
ans = re.findall('[A-Z]', s)
print(*ans, sep='')


'''
問題A_401：Status Code
'''

stats_code = int(input())
print('Success' if 200 <= stats_code <= 299 else 'Failure')

'''
問題A_400：ABC400 Party
'''
n = int(input())
res = -1
if 400 % n == 0:
  res = 400 // n
print(res)

'''
問題A_399：Hamming Distance
ジェネレーター式
'''
n = int(input())
s,t = input(), input(); 
print(sum(1 for i in range(n) if s[i] != t[i]))

'''
問題A_398：Doors in the Center
'''
n = int(input())
ans = ['-'] * n; mid = (0 + n) // 2
for _ in range(n):
  if n % 2 == 0:
    ans[mid-1:mid+1] = '='*2
  else:
    ans[mid] = '='
  break
print(*ans, sep='')

'''
問題A_397：Thermometer
'''

x = float(input())
type = 0
if x >= 38.0:
  type = 1
elif 37.5 <= x < 38.0:
  type = 2
else:
  type = 3
print(type)

'''
問題A_396：Triple Four・・・解けなかった

方針：
先頭から順に見ていき、暫定の整数が一致しない場合は
暫定の整数を更新
一致する場合は、件数をカウントし、件数が3件だった場合は処理を終了する
'''

n = int(input())
num = list(map(int, input().split()))
for i in range(n -1):
  if num[i] == num[i + 1] == num[i + 2]:
    print('Yes')
    exit()
print('No')

'''
問題A_395：Strictly Increasing?

A1 < A2 < A3

方針：
全探索
Ai ≦ Ai + 1が成り立つならYes
成り立たないならNoで処理を終了
'''
n = int(input())
num = list(map(int, input().split()))
for i in range(n - 1):
  if num[i + 1] <= num[i]:
    print('No')
    exit()
print('Yes')

'''
問題A_394：22222
'''
s = list(input())
ans = [s_n for s_n in s if s_n == '2']
print(''.join(ans))

# 別解
# リストに含まれている'2'の件数を取得し、
# '2'*cする 22222...になる
s = list(input())
c = s.count('2')
print('2'*c)

'''
問題A_393： Poisonous Oyster

sick sick = 1
sick fine = 2
fine sick = 3
fine fine = 4

tさんがsickかfineか
'''

t,a = input().split(); ans = 0
ans = 1 if t == 'sick' and  a == 'sick' else 2
ans = 3 if t == 'fine' and a == 'sick' else 4
print(ans)

'''
問題A_392：Shuffled Equation
'''

def is_solve(a: int, b: int, c: int) -> bool:
  if a * b == c:
    return True
  if a * c == b:
    return True
  if b * c == a:
    return True
  return False

n1, n2, n3 = map(int, input().split())
print('Yes' if is_solve(n1, n2, n3) else 'No')

'''
問題A_391：Lucky Direction

方針：
1文字ずつ取り出して文字列連結する

別解
辞書にパターン分のデータを用意
'''

d = input(); ans = ''
for c in d:
  if c == 'N':
    ans += 'S'
  if c == 'S':
    ans += 'N'
  if c == 'E':
    ans += 'W'
  if c == 'W':
    ans += 'E'
print(ans)

'''
問題A_390：12345

時間計算量：O(N)

方針
次の要素の方が前の要素より小さい時のみ値を1回だけ入れ替えて並べ替える
入れ替えた回数をカウントし処理終了
'''

n = list(map(int, input().split())); cnt = 0
target = [1 ,2, 3, 4, 5]; ok = False
for i in range(1, len(n)):
  if n[i] < n[i - 1]:
    n[i], n[i - 1] = n[i - 1], n[i]
    ok = True
    break
print('Yes' if ok and  n == target else 'No')

'''
問題A_389：9x9
'''

n1, s, n2 = list(input())
print(int(n1) * int(n2))

'''
問題A_388：
'''
s = input()
print(s[0] + 'UPC')


'''
問題A_387：Happy New Year 2025
'''






