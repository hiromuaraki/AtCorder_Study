from itertools import combinations

'''
全104問
[書籍]問題解決のためのアルゴリズム×数学が基礎からしっかりに身につく本のもんdない

計算量の考え方；ざっくりとオーダー記号で表したPCでの計算回数
一般家庭のPCの目安＝10^9回程度（実行時間2秒以内）
'''

'''
問題0_1-100までの総和
単純に N - 1回の計算が発生する
N=100 100-1 =99回

等差数列
隣りあった数の差が同じ
(初項 + N) * 数列の数 / 2

合計101となる50個のペア
101*50=5050


整数：小数点がつかない数：0, -1, 10（分数は除く）
有理数：整数/整数で表すことのできる数・・・0, -1, 1.0, 10、2/3
実数：数直線上で表すことのできる数・・・・π, 0, -1, 1.0, 10、2/3, √3
正の数：0より大きい数
負の数：0未満の数


指数関数：
a^m * a^n = a^m+n
2^5*2^4 = 2^9

a^m/a^n = a^m-n
2^9/2^5 = 2^4

(a^m)^n = a^mn
(2^5)^3 = 2^5*2^5*2^5 = 2^15

a^mb^m = (ab)^m
2^5*3^5 = 6^5
'''
print((1 + 100)*100//2)

'''
3 3
Σ Σ ij
i=1j=1

(1*1)*(1*2)+(1*3)+(2*1)+(2*2)+(2*3)+(3*1)+(3*2)+(3*3)
'''
result = 0
for i in range(1,4):
  for j in range(1, 4):
    result += (i*j)
print(result)


'''
10進法から2進法へ変換

例）10の場合→1010
10 % 2 = 0
 10 // 2 = 5
5 % 2 = 1
 5 // 2 = 2
2 % 2 = 0
 2 // 2 = 1
1 % 2 = 1
 1 // 2 = 0・・・ここでループ終了

0101→1010

割った余りを記録
割った商を更新
Nが0になるまで割り続ける
最後に下から順に読む
'''

N = int(input())
ans = ''
while N >= 1:
  if N % 2 == 0: ans += '0'
  if N % 2 == 1: ans += '1'
  N //= 2
print(ans[::-1])

'''
問題001_Print 5+N（文字式）
'''

print(int(input()) + 5) 

'''
問題002_Sum of 3 Integers（足し算）
'''

A,B,C = map(int, input().split())
print(A+B+C)

'''
問題003_Sum of N Integers（足し算）
'''
N = int(input())
print(sum(list(map(int, input().split()))))


'''
問題004_Product of 3 Integers（掛け算）
'''

A,B,C = map(int, input().split())
print(A*B*C)

'''
問題005_Modulo 100（剰余）
'''

N = int(input())
A = list(map(int, input().split()))
print(sum(A) % 100)


'''
問題006_Print 2N+3（１次関数）
時間計算量：O(1)
'''
N = int(input())
print(2+N+3)



# 計算時間の見積もり
'''
問題007_Number of Multiples 1（線形時間）・・・考え方の整理
時間計算量の問題
時間計算量：O(N）

→(N+1)/2
'''

N,X,Y = map(int, input().split())
M = [X, Y]; ans,i = 0, 1
for i in range(1, N+1):
  if i % X == 0 or i % Y == 0: # 倍数の判定
    ans += 1
print(ans)

'''
問題008_Brute Force 1（全探索）・・・解けた

時間計算量：O(N^2)

1 ≦ N ≦ 1000
N^2通り N*N = 1000*1000 = 1000000 = 10^6回
カードの合計がS以下になる書き方を求める（X）
N=3, S=4の場合
4以下になる組合せは
(1,1)(1,2)(1,3)
(2,1)(2,2)
(3,1)
6通り
'''
ans = 0
N,S = map(int, input().split())
for i in range(1, N+1):
  for j in range(1, N+1):
    if i+j <= S:
      ans += 1
print(ans)

'''
問題009_Brute Force 2（組合せの全探索）・・・解けない（土日やる） 

カードの枚数 N枚
2^N（指数関数）に回増えていく

2^N-1回ループ
P66に記載
1.2進法を利用して選び方に番号を振る
2.選び方の番号を全探索

'''

'''
問題010＿Brute Force 2（階乗）
階乗
N=5
5! = 1*2*3*4*5

時間計算量：O(N)
'''
N = int(input())
ans = 1
for i in range(1, N+1):
  ans *= i
print(ans)

# 再帰処理の場合
def f(n: int):
  if n == 1:
    return 1
  return n*f(n - 1)

'''
問題011_Print Prime Numbers（素数）・・・解けない（土日やる）
N以下の素数を出力する

2..N+1回繰り返す

1と自分自身以外に約数を持たない＝約数（割り切れる数が2個）
'''


N = int(input())
ans = []
def isprime(n :int) -> bool:
  for i in range(2, n):
    if n % i == 0:
      return False
  return True

for i in range(2, N+1):
  # 素数判定
  if isprime(i):
    ans.append(i)
print(*ans)

'''
問題012_Primality Test・・・理解した

時間計算量：O(√N)

素数判定
1..N-1通りまで調べない

方針：
2..√Nまでの範囲で割り切れなければ素数と判定
2,3,5,7→4,9,25,49
'''

N = int(input())
for i in range(2, int(N**0.5)+1):
  if N % i == 0:
    print('No')
    exit()
print('Yes')



'''
問題013_Divisor Enumeration（約数列挙）・・・理解できた

計算量を工夫したプログラムにしてください
順序は問わない
時間計算量：O(√N)

方針：
1..√N回繰り返す
Nがiで割れたらiを出力
N//iとiが違うならN//iを出力
'''

N = int(input())
for i in range(1, int(N**0.5)+1):
  if N % i != 0:
    continue
  print(i)
  n = N // i
  if i != n:
    print(n)

'''
問題014_Factorization（素因数分解）
時間計算量：O(√N lon N)
N**0.5→ルートに変換している
√Nを計算している
0.5＝1/2つまり、ルートを意味する

2..√N回繰り返す
Nを2で割れるだけ割る
Nを3で割れるだけ割る
最後にNが1以外の場合、それを素因数に加える
'''

N = int(input())
ans = []
for i in range(2, int(N**0.5)+1):
  while N % i == 0:
    N //= i
    ans.append(i)
if N >= 2:
  ans.append(N)
print(*ans)


'''
問題015_Calculate GCD（最大公約数）
ユークリッドの互除法
＊再帰処理

時間計算量：O(log(A+B))

大きい数から小さい数で割った余りを書き換える操作を繰り返す
0になった時点で割った数が最大公約数

注意）
2回目のループ以降x,yの値が入れ替わる
'''

A,B = map(int, input().split())
if A < B:
  A,B = B, A

def gcd(x: int, y: int) -> int:
  if x % y == 0:
    return y
  return gcd(y, x % y)
print(gcd(A, B))

'''
問題016_Greatest Common Divisor of N Integers（最大公約数）・・・解けた
ユークリッドの互除法

2つの最大小約数を計算し次の最大公約数を求めることを繰り返す
前の計算結果と以降の数列の最大小約数を求めることをN-2回繰り返す
'''

N = int(input())
A = list(map(int, input().split()))

def calc_gcd(x: int, y: int) -> int:
  if x % y == 0:
    return y
  return calc_gcd(y, x % y)

x,y = A[0],A[1]
if  x < y:
  x, y = y ,x
# 1回目の最大公約数を求める
ans = calc_gcd(x, y)

for i in range(N-2):
  # 最大公約数を求める
  ans = calc_gcd(A[i+2],ans)
print(ans)


'''
問題017_Least Common Multiple of N Integers（最小公倍数）・・・解けた
1回目の最大公約数を求める
Ai * Ai+1 / 最大公約数 = 最小公倍数
1回目の結果を元に同じ処理をN-2回繰り返す
'''

N = int(input())
A = list(map(int, input().split()))

# 最大公約数を求める
def gcd(x: int, y: int) -> int:
  if x % y == 0:
    return y
  return gcd(y, x % y)

# 最小公倍数を求める
# 1つ目の数 * 2つ目の数 / 最大公約数
def lcm(x: int, y: int, gcd: int) -> int:
  return (x * y) // gcd

x,y = A[0],A[1]
if x < y:
  x,y = y,x
ans = lcm(A[0], A[1], gcd(x, y))

for i in range(N-2):
  ans = (A[i+2] * ans) // gcd(A[i+2], ans)
print(ans)


'''
問題018_Convenience Store 1（場合の数）・・・理解できた
解答の意味がわからない

場合分け
合計が500円になる組み合わせは
100+400 = 500
200+300 = 500 のみ

100,200,300,400をa,b,c,dとしそれぞれのコインの値段の枚数を数える
a*d = b*c
ac+bcが求める答え

コインの枚数をcount関数でカウントしている

500円になる組合せ
100+400=500
200+300=500

100,200,300,400の4種類
a(100),b(200),c(300),d(400)

a*d + b*c = 500

コインの枚数の分布を取る
100*iで枚数に変換

5
100 300 400 400 200

100*1=100
100*2=200
100*3=300
100*4=400

100, 200, 300, 400
1    1    1    2

1*2+1*1=3
'''

N = int(input())
A = list(map(int, input().split()))

a,b,c,d = 0,0,0,0
for i in range(N):
  if A[i] == 100:
    a += 1
  if A[i] == 200:
    b += 1
  if A[i] == 300:
    c += 1
  if A[i] == 400:
    d += 1
print(a*d+b*c)

# 下のコードはO(N^2)でTLE
# coin = [A.count(100*i) for i in range(1, N)]
# print(coin[0]*coin[3]+coin[1]*coin[2])  

'''
問題019_Choose Cards 1・・・解けた（場合の数）（要復習）

左から順に同じ色のカードを２枚選ぶ場合の数
x, y, z

各色のカードの枚数のs分布を取る

xC2+yC2+zC2通りの場合の数
＝x(x-1)/2+y(y-1)/2+z(z-1)/2
'''

N = int(input())
A = list(map(int, input().split()))
x, y, z = 0,0,0
for i in range(N):
  if A[i] == 1:
    x += 1
  if A[i] == 2:
    y += 1
  if A[i] == 3:
    z += 1

x = x*(x-1)//2
y = y*(y-1)//2
z = z*(z-1)//2
print(x+y+z)


'''
問題020_Choose Cards 2（全探索）・・・（土日やる）
nC5の組み合わせ ５枚選ぶ
＊異なるn個のものの中からr個選ぶ

時間計算量：O(N^4)
4枚のカードの番号が決まれば残りのカードの番号も分かる
4重ループで実装できる。

Ai+1+Ai+2+Ai+3*Ai+4+Ai+5 = x
Ai+5 = x-Ai+1-Ai+2-Ai+3-Ai+4 
1000 - Ai+1 - Ai+2 - Ai+3+Ai+4 = Ai+5

x = 1000 - (i+j+k+l)

だたし
・同じカードを2回使ってはいけない
・同じ組み合わせの重複のカウントを防ぐ    i < j < k < l < m
'''
N = int(input())
A = list(map(int, input().split()))

# from itertools import combinationで5枚の組合せを全て取り出す場合
# ans = 0
# for cards in combinations(A, 5):
#   if sum(cards) == 1000:
#     ans += 1
# print(ans)

N = int(input())
A = list(map(int, input().split()))
A_set = set(A)
ans = 0
for i in range(N):
  for j in range(i+1, N):
    for k in range(j+1, N):
      for l in range(k+1, N):
        x = 1000 - (A[i]+A[j]+A[k]+A[l])
        for m in range(l+1, N):
          if A[m] == x:
            ans += 1
print(ans)




'''
問題021_Combination Easy（nCr）・・・解けない（理解できた）(土日復習)
nCr = N!/(r!*(n-r)!)

例）N=6 r=2
6!/2!*(6-2)!
=6!/2!*4!
=720/48
=15

N!を求める

r!を求める

N-r!を求める

上記の公式で計算
n!/r*(n-r)!

nPr = r!*nCr

nCr・・・順番を区別しない
nPr・・・順番を区別する
'''

n,r = map(int, input().split())
# 階乗の計算
# nCr = n!/r!*(n-r)!
# N!を求める
fact_n = 1
for i in range(1, n+1):
  fact_n *= i

# r!を求める
fact_r = 1
for i in range(1, r+1):
  fact_r *= i

# n-r!を求める
fact_nr = 1
for i in range(1, n-r+1):
  fact_nr *= i

combination = int(fact_n / (fact_r*fact_nr)) # nCr
print(combination)
print(fact_r * combination) # nPr

'''
問題022_Choose Cards 3・・・解けない（要復習）

出目の和期待値＝（青の出目の期待値）＋（赤の出目の期待値）
青の出目の期待値＝（B1+B2+Bn）/ N
赤の出目の期待値＝（R1+R2+Rn）/ N

（B1+B2+Bn）/ N ＋（R1+R2+Rn）/ N
'''

'''
問題023_Dice Expectation・・・（期待値）・・・解けた
＊和の期待値は和となる性質

2つの試行を行う

1番目の試行をX、2番目の試行をYとする
Xの期待値をE[X], Yの期待値をE[Y]とするとき、
X＋Yの期待値 = E[X]+E[Y]

2つの出目の和の期待値＝ (X1+X2+X3..Xn / N) + (Y1+Y2+Y3...Yn) / N

:.12f・・・小数点第12位まで出力するフォーマット指定子
'''


N = int(input())
B = list(map(int, input().split()))
R = list(map(int, input().split()))
print('{:.12f}'.format(sum(B) / N + sum(R) / N))

'''
問題024_Answer Exam Randomly・・・（期待値）・・解けた

合計点数の期待値＝(1問目の点数の期待値) + (2問目の点数の期待値)+...(N問目の点数の期待値)

合計点数の期待値＝Q1/P1 + Q2/P2...+Qn/Pn
'''

N = int(input())
ans = 0.0
for i in range(N):
  P,Q = map(int, input().split())
  ans += Q / P
print('{:.12f}'.format(ans))

'''
問題025_Jiro's Vacation（期待値）・・・解けない

サイコロの確率 1/6

1,2の場合：
1回目が1の目が出る確率・・・1/6
2回目が2の目が出る確率・・・1/6
＊1/6 + 1/6 = 2/6 = 1/3

それ以外の場合（3,4,5,6）
＊4/6 = 2/3
'''

N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
ans = 0.0
for i in range(N):
  ans += A[i]*(1.0 / 3.0) + B[i]*(2.0 / 3.0)
print('{:.12f}'.format(ans))

'''
問題026_Coin Gacha・・・解けない （保留）
'''





'''
動的計画法028_Frog 1・・・解けない（動的計画法）

時間計算量：O(V＋E）
V＝ノード数 E = エッジ数

動的計画法とは＝計算の重複を避けるアルゴリズム（無駄をなくす）

動的計画法のイメージ（以下の３つの特徴を持ったアルゴリズム）
・問題をいくつかの小さな問題に分割
・それぞれの問題の計算結果を表に記録
・同じ問題に対して表から計算結果を参照する

大きな問題を小さな問題に分割し最終的な解を求める
→分割統治法

＊計算の重複を避ける（メモ化）

漸化式
𝐹⁡(𝑛)=𝐹⁡(𝑛−1)+𝐹⁡(𝑛−2)𝐹⁡(0)=0,𝐹⁡(1)=1

4
10 30 40 20


フィナボッチ数列：
f(n) = f(n - 1) + f(n - 2)
a1,a2,a3..an
a1+a2 = a3
i,i+1,i+2...i+N
i+i+1 = 2i+1
'''

'''
||：絶対値
[何かしら整数]：切り捨て

足場１から足場へ移動するコスト＝０
足場１かあ足場iへ移動するコスト＝|i - i+1|
足場１から足場２へ移動するコスト＝|i - i+2|

足場i=1..7まである

N = 7
A = 2,9,4,5,1.6.10

開始
[0, None,None,None,None,None,None]

i = 1
dp[0] + |2-9| = 7
[0, 7,None,None,None,None,None]

i = 2..7まで繰り返す
ノード2から来る方法
dp[i - 1] + |9-4| = 12

ノード1から一つ飛ばして来る方法
dp[i - 2] + |2-4| = 2

最小値の比較（min）
最小値は2だから、2をdpに記録
[0, 7, 2,None,None,None,None]

i = 3
ノード3から来る方法
dp[i - 1] + |4-5| = 3

ノード2から一つ飛ばして来る方法
dp[i - 2] + |9-5| = 11

最小値の3を記録
[0, 7, 2, 3,None,None,None]

i = 4
ノード4から来る方法
dp[i - 1] + |5-1| = 7

ノード3から一つ飛ばして来る方法
d@[i - 2] + |4-1| = 5

最小値の3を記録
A = 2,9,4,5,1.6.10
[0, 7, 2, 3, 5,None,None]

i = 5
ノード5から来る方法
dp[i - 1] + |1-6| = 10

ノード4から一つ飛ばして来る方法
d@[i - 2] + |5-6| = 4

最小値の4を記録
[0, 7, 2, 3, 5, 4,None]

i - 6
ノード6から来る方法
dp[i - 1] + |6-10| = 8

ノード5から来る方法
dp[i - 2] + |1-10| = 14

最小値の8を記録
[0, 7, 2, 3, 5, 4, 8]

最小回数＝８
'''

# 他の人の解法
N = int(input())
A = list(map(int, input().split()))
dp = [0]*N
# 動的計画法
for i in range(1, N):
  if i == 1:
    dp[i] = dp[i - 1] + abs(A[i - 1] - A[i])
    continue
  dp[i] = min(dp[i - 1] + abs(A[i - 1] - A[i]),
              dp[i - 2] + abs(A[i - 2] - A[i]))
print(dp[N - 1])


'''
問題029_Climb Stairs・・・解けた（動的計画法）

dp[n] = dp[n - 1] + dp[n - 2]
→フィナボッチ数列
'''

N = int(input())
dp = [0]*(N+1)
dp[0], dp[1] = 1, 1
for i in range(2, N+1):
  dp[i] = dp[i-1]+dp[i-2]
print(dp[N])


'''
問題031_Taro's Vacation・・・動的計画法・・・解けない（要復習）
計算量の工夫の問題
'''
# これはダメ
N = int(input())
A = list(map(int, input().split()))
m = 0
for i in range(N-2):
  if m < A[i] + A[i+2]:
    m = A[i] + A[i+2]
print(m)

'''
問題032_Binary Search（二分探索）・・・解けた
時間計算量：O(log N）

あらかじめ昇順または降順のデータに対して使うアルゴリズム
'''

N, X = map(int, input().split())
A = list(map(int, input().split()))
A.sort()
left, right = 0, len(A)-1
ans = 'No'
while left <= right:
  mid = (left + right) // 2
  if X == A[mid]:
    ans = 'Yes'
    break
  if A[mid] < X:
    left = mid + 1
  else:
    right = mid - 1
print(ans)






'''
問038_How Many Guests? ・・・解けない・・・階差と累積和（要復習）
時間計算量：O(N±Q）

階差：Bi = Ai-Ai- 1
累積和：Bi = A1+A2+...Ai
累積和ー階差
'''

# TLE（実行制限時間エラー）
# 計算回数が重複しているから改善できる
N,Q = map(int, input().split())
A = list(map(int, input().split()))
for i in range(Q):
  L, R = map(int, input().split())
  print(sum(A[L-1:R]))