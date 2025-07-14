# 1章から3章までの理解度チェック

'''
再度復習が必要

2章
・関数（指数関数、対数関数）
・ビット演算
・＊時間計算量

３章
・確率（期待値）
・＊全探索
・＊ソート
・＊再帰
・＊漸化式
・＊動的計画法（DP）

４章
・階差と累積和

'''

'''
一般家庭用のPCの時間計算量
ざっくり10^9回（実行時間１秒）
1000000000(1秒間に10億回の処理が可能)
'''

'''
10進法→2進法・・・・・・・解けた
kを変更すればN進法に対応可能

b=2進法
桁の位：1, 2, 4, 8, 16, 32, 64, 128....N

d=10進法
0,1,2,3,4,5,6,7,8,9

x=16進法
1,2,3,4,5,6,7,8,9,A,B,C,D,E,F,G

'''
N = int(input())
k = 2
ans = ''
while N > 0:
  ans += str(N % k)
  N //= k
print(ans[::-1])

'''
二分探索法

＊前、後ろに探索範囲を絞る際にpivot(中央値の数)を加算または減算するのが重要
'''

N = 10
A = list(range(1, 11))
left, right = 0, len(A)-1
pivot = (left + right)//2 # 基準値
x = int(input('1-20までの整数を1つ入力してください。'))
while left <= right:
  pivot = (left + right)//2
  if A[pivot] == x:
    break
  if A[pivot] < x:
    left = pivot+1
  else:
    right = pivot-1

print(f'検索の数値{x}が{pivot}番目に見つかりました。')

'''
クイックソート（再帰処理）

データが昇順または降順である必要がある
中央値を決め大小のグループに分け並べ替える
'''

def quick_sort(arr: list) -> list:
  if len(arr) <=1:
    return arr
  
  pivot = arr[len(arr)//2]
  left = [x for x in arr if x < pivot]
  mid  = [x for x in arr if x == pivot]
  right = [x for x in arr if x > pivot]
  return quick_sort(left) + mid + quick_sort(right)

A = [3, 7, 10, 1, 0, 2, 3, 20, 1, 4, 9, 13, 15]
print(quick_sort(A))

'''
問題004_Product of 3 Integers・・・文字式・・・解けた
'''
A1,A2,A3 = map(int, input().split())
print(A1*A2*A3)

'''
問題005_Modulo 100・・・剰余・・・解けた
'''

N = int(input())
A = list(map(int, input().split()))
print(sum(A) % 100)

# 別解
total = 0
for a in A:
  total += a
print(total % 100)


'''
問題006_Print 2N+3・・・定数時間・・・解けた
時間計算量：O(１）・・・１通り
'''

N = int(input())
print(2*N+3)

'''
問題007_Number of Multiples 1・・・線形探索・・・解けた

時間計算量：O(N)・・・N通り
(N+1)/2
'''

N,X,Y = map(int, input().split())
ans = 0
for i in range(1,N+1):
  if i % X == 0 or i % Y == 0:
    ans += 1
print(ans)

'''
問題008_Brute Force 1・・・全探索・・・解けた
時間計算量：O(N^2）・・・N^2通り
N(N-1)/2
'''

N, S = map(int, input().split())
ans = 0
for i in range(1, N+1):
  for j in range(1, N+1):
    if i+j <= S:
      ans += 1
print(ans)

'''
問題009_Brute Force 2・・・全探索＆動的計画法（土日やる）要復習・・・解けない＊
全探索で解いた場合：2^N回（指数関数）
＊動的計画法を勉強す
'''
# 他の人の解法
N, S = map(int, input().split())
A = list(map(int, input().split()))
dp = [False]*(S+1)
dp[0] = True

for An in A:
  for i in range(S, An - 1, -1):
    if dp[i -An]:
      dp[i] = True

if dp[S]:
  print('Yes')
else:
  print('No')

'''
問題010_Factorial・・・階乗・・・解けた

N!
N*(N-1)*(N=2)....1*2*3
'''

N = int(input())
ans = 1
for i in range(1, N+1):
  ans *= i
print(ans)


'''
①数学のお勉強

素数・・・1と自分自身でしか割り切れない数（他の合成数で割れない）
2,3,5,7,11....
2..√N回の計算量
エラトステネスのふるいのアルゴリズムがある

約数・・・1以上の自然数で割り切れる数
例）12の約数
1,2,3,4,6,12
1..N回の計算量
割った余り、割れた商が約数

公約数・・・・2つ以上の正の約数に共通する約数
例）6,4の場合 = 2
4: 1,2,4
6: 1,2,3,6

公倍数・・・2つ以上の正の倍数に共通する倍数

最大公約数・・・・公約数の中の一番大きい数
時間計算量：O(N))
xとyが割り切れるまで繰り返す。割り切れたらyを返す
ユークリッドの互除法

最小公倍数・・・公倍数の中の一番小さい数
例）12,24の場合 = 24
12: 12, 24, 36 48 60, 72, 84, 96 108, 120
24: 24......
1つ目の数＊2つ目の数/最大公約数＝最小公倍数


等差数列・・・・隣り合った数の差が同じ
並べ方：
a,b,c：b - a = c - b
a,c,b：c - a = b - c
b,c,a：c - b = a - c

等比数列・・・・前と後ろの割った比が同じ
a,b,c,d

b/a = d/c
b/a = c/b
c/a = d/b



集合
U・・・全体集合
A・・・集合A
B・・・集合B
空集合・・・{}
x∈A・・・集合Aに要素xが含まれる
A∩B・・・積集合（共通部分）キャップ
A∪B・・・和集合（A＋Bを足したもの）カップ・・・集合A,Bのうち少なくとも一方に含まれる要素
A＋B＋共通
A⊂B・・・集合Aの要素がすべて集合Bに含まれ値る（集合Aは集合Bの部分集合）

Python
積集合：＆・・・A,Bの共通部分
和集合：＋・・・A,Bのうち少なくとも一方に含まれる要素を足した要素
差集合：ー・・・A,Bのうち一方にしか含まれない要素
対称差：^・・・A,Bの共通部分以外の要素

'''


'''
問題011_Print Prime Numbers・・・素数・・・解けた

N以下の素数の出力
2..N回繰り返す
'''

N = int(input())

def isprime(n):
  for i in range(2, n):
    if n % i == 0:
      return False
  return True

for i in range(2, N+1):
  if isprime(i):
    print(i, end=' ')


'''
問題012_Primality Test・・・素数・・・解けた
2..√N回繰り返す
時間計算量：O(√N）
'''

N = int(input())
for i in range(2, int(N**0.5)):
  if N % i == 0:
    print('No')
    exit()
print('Yes')

'''
問題013_Divisor Enumeration・・・約数・・・解けた
1..√N回繰り返す

時間計算量：O(√N）
'''

N = int(input())
ans = []
for i in range(1, int(N**0.5)+1):
  if N % i != 0:
    continue
  ans.append(i)
  if i != N // i:
    ans.append(N // i)
print(*sorted(ans))

'''
問題014_Factorization・・・素因数分解・・・解けた
2..N回繰り返す

時間計算量：O(√N）
'''

N = int(input())
ans = []
for i in range(2, N+1):
  while N % i == 0:
    ans.append(i)
    N //= i
print(*ans)


'''
問題015_Calculate GCD・・・最大公約数(2つ)・・・解けた
時間計算量：O(log(A+B)))

ユークリッドの互除法
x % y = あまり
y // あまり

大きい方の数を「大きい方を小さい方で割った余り」に書き換えるという操作を繰り返す
割り切れるまで繰り返す
割り切れたらyを返却しプログラム修了
'''

X, Y = map(int, input().split())
if X < Y:
  X, Y = Y, X
def calc_gcd(x: int , y: int) -> int:
  if x % y == 0:
    return y
  return calc_gcd(y, x % y)

print(calc_gcd(X, Y))

'''
問題016_Greatest Common Divisor of N Integers・・・最大公約数（複数）・・・解けた

はじめに１組の最大公約数を求める
2組以降は求めた最大公約数と比較する
'''

N = int(input())
A = list(map(int, input().split()))

def calc_gcd(x: int, y: int) -> int:
  if x % y == 0:
    return y
  return calc_gcd(y, x % y)

# 1組のペアの最大公約数を求める
x, y = A[0], A[1]
if x < y:
  x, y = y, x

ans = calc_gcd(x, y)
for i in range(2, N):
  ans = calc_gcd(ans, A[i])
print(ans)

'''
問題017_Least Common Multiple of N Integers・・・最小公倍数・・・解けた

1つ目の数＊2つ目の数/最大公雨約数＝最小公倍数
という性質がある
'''

N = int(input())
A = list(map(int, input().split()))

def gcd(x: int, y: int) -> int:
  if x % y == 0:
    return y
  return gcd(y, x % y)

x,y = A[0],A[1]
if x < y:
  x, y = y,x
# 1組目のペアの最小公倍数をを先に求める
ans = (x * y) // gcd(x, y)

for i in range(2, N):
  ans = (ans * A[i]) // gcd(ans, A[i])
print(ans)

'''
②数学のお勉強

＊関数
y=x（比例）直線・・・原点を通る
y=x/a（反比例）双曲線
y=x^2（2次関数）放物線

y=ax+b・・・一次関数（直線）
a＝傾き b＝切片（y軸方向にb移動する）
b=0でなければ原点を通らない

y=ax^2+bx+c・・・2次関数

y=数値^x・・・指数関数（xが指数）
右肩上がりに急激に増加

<指数法則>

公式１
a^m * a^n = a^m+n
2^5 * 2^4 = 2^9

公式２
a^m / a^n = a^m-n
2^9 / 2^5 = 2^9-5 = 2^4

公式３
(a^m)^n = a^mn
(2^5)^3 = 2^5*3 = 2^15

公式４
a^mb^m = (ab)^m
2^5 * 3^5 = (2*3)^5*5 = 6^5

対数関数・・・・y = log数値 X・・・aを何乗したらXになるか
a = 底  b = 真数
右肩上がりに緩やかに増加

log = 何乗したら目的の数になるかを表した記号

loga b・・・・aを何乗したらbになるか
log10 X・・・常用対数
log2 X・・・二分探索法

例）
10^3 = 1000
log10 1000

2^0 = 1
log2 1 = 0

2^1 = 2
log2 2 = 1

2^2 = 4
log2 4 = 2

2^3 = 8
log2 8 = 3

2^4 = 16
lgo2 16 = 4

2^5 = 32
log2 32 = 5
'''

'''
問題018_Convenience Store 1・・・場合の数（頻度分布）・・・解けた

100,200,300,400
a,b,c,d
a*d = b*c
100+400=500
200+300=500

それぞれのコインの枚数の頻度分布を取る
a*d+b*c=合計500円になるコインの枚数
'''

N = int(input())
A = list(map(int, input().split()))
a,b,c,d = 0,0,0,0
for coin in A:
  if coin == 100:
    a += 1
  if coin == 200:
    b += 1
  if coin == 300:
    c += 1
  if coin == 400:
    d += 1
print(a*d + b*c)

'''
問題019_Choose Cards 1・・・場合の数（組合せ）・・・解けた

N枚のカードの中から同色のカードを２枚選ぶ

赤=x、青=y、黄=zのカードが３枚

xC2通りある
xC2/2 + yC2/2 + zC2/2

答え：-1は3枚から自分を除いている
=x(x-1)//2 + y(y-1)//2 + z(z-1)//2
'''

N = int(input())
A = list(map(int, input().split()))
x,y,z = 0,0,0
for color in A:
  if color == 1:
    x += 1
  if color == 2:
    y += 1
  if color == 3:
    z += 1
print(x*(x-1)//2 + y*(y-1)//2 + z*(z-1)//2)

'''
問題020_Choose Cards 2・・・場合の数（全探索）・・・解けた

時間計算量：O(N^5)

rangeの探索範囲が変化していくのが肝
'''

N = int(input())
A = list(map(int, input().split()))
ans = 0
for a in range(N):# 0
  for b in range(a+1, N): # 1
    for c in range(b+1, N): # 2
      for d in range(c+1, N): # 3
        for e in range(d+1, N): # 4
          if A[a]+A[b]+A[c]+A[d]+A[e] == 1000:
            ans += 1
print(ans)

'''
問題021_Combination Easy・・・組合せ（nCr）・・・解けた

n! / r!(n-r)!

n!を求める
r!を求める
n-r!を求める
n! / r!*(n-r)!をする
'''

n,r = map(int, input().split())

# n!
fact_n = 1
for i in range(1, n+1):
  fact_n *= i

# r!
fact_r = 1
for i in range(1, r+1):
  fact_r *= i

# n-r!
fact_nr = 1
for i in range(1, n-r+1):
  fact_nr *= i

print(fact_n // (fact_r * fact_nr))


'''
問題022_Choose Cards 3・・・全探索・・・解けない（要復習）

計算量の工夫が必要
'''

'''
問題023_Dice Expectation・・・確率（期待値、等確率）・・・解けた
賞金の期待値を求める
期待値の線形性

出目の和の期待値＝（青の出目の期待値）＋（赤の出目の期待値）
出目の期待値＝（A1+A2+....AN）// N

１回目のサイコロの試行・・・E[x]
２回目のサイコロの試行・・・E[y]
E[x]+E[y]＝２つのサイコロの和の期待値
'''

N = int(input())
B = list(map(int, input().split()))
R = list(map(int, input().split()))
print('{:.12f}'.format(sum(B)//N + sum(R) // N))

'''
問題_024_Answer Exam Randomly・・・確率（期待値、ランダム）・・・解けた

合計点数の期待値＝（1問目の期待値）＋（N問目の期待値）

Q1/P1+Q2/P2=期待値
'''

N = int(input())
ans = 0
for i in range(N):
  p1, p2 = map(int, input().split())
  ans += p2/p1
print('{:.12f}'.format(ans))

'''
問題025_Jiro's Vacation・・・確率（期待値）・・・解けた

サイコロを振り1-6が出る確率：1/6
1,2が出る確率：1/6+1/6 = 2/6 約分し、1/3
3.4.5.6が出る確率：1 - 1/3 = 2/3
'''


N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
print(sum(A) * 1/3 + sum(B) * 2/3)

'''
問題026_Coin Gacha・・・確率（期待値）・・・解けない（要復習）
'''

'''
③数学のお勉強

漸化式・・・前の項の結果からその項の値を求める規則のこと・・・an = 2n - 1
フィナボッチ数列：前の２つの項を足した値
'''

'''
問題028_Frog 1・・・動的計画法（漸化式）・・・解けない（要復習）
時間計算量：O(N）

dp[N]番目が答え

カエルが足場Nまでにたどりつくまでに支払うコストの総和の最小値を求める

必要な情報：
i番目に支払ったコスト(dpテーブルを作る)

コストの計算
1通り
足場1-1：・・・dp[0] = 0

足場1-2：・・・dp[1] = dp[0] + |足場2-足場1|

2通り
足場1-3
  足場2-3： d[[2] = dp[1] + |足場3-足場2|
  足場1-3： d[[2] = dp[0] + |足場3-足場1|

足場2-4
  足場3-4： dp[3] = dp[2] + |足場4-足場3|
  足場2-4： d[[3] = dp[1] + |足場4-足場2|

足場3-5
  足場4-5：dp[4] = dp[3] + |足場5-足場4|
  足場3-5：dp[4] = dp[2] + |足場5-足場3|
'''

N = int(input())
A = list(map(int, input().split()))
dp = [0]*N
for i in range(1, N):
  if i == 1:
    dp[i] = dp[i-1] + abs(A[i]-A[i-1])
    continue
  dp[i] = min(
    dp[i-1] + abs(A[i-1] - A[i]),
    dp[i-2] + abs(A[i-2] - A[i]))
  
print(dp[N-1])


'''
問題029_Climb Stairs・・・動的計画法（漸化式）・・・解けた
0段目からN段目までに辿り着くまでの移動方法が何通りあるか

0-0段目：1
0-1段目：1

2段目以降は
n-1段目から1歩で1段上がる
n-2段目から2歩で2段上がる

フィナボッチ数列＝
dp[N] = dp[N-1]+dp[N-2]

求める答え：
dp[N] = dp[i-1]+dp[i-2]
'''




