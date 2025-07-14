'''
問題B_414_String Too Long・・・ランレングス圧縮（文字列操作）・・・解けなかったが自力後後日AC
難易度37

c1,c2,c3..cnを連結した文字列Sを出力
ただし|S| > 100場合は、’Too Long’

|S| <= 100：S
100を超える場合は、’Too Long’を出力

Li ≦ 10^18（18桁）→掛け算するとメモリエラーになり例外で落ちる
'''
N = int(input())
MAX_LENGTH = 10**6; s = ''
for _ in range(N):
  C, L = input().split()
  L = int(L)
  if L > MAX_LENGTH:
    print('Too Long')
    exit()

  if len(s) > 100:
    print('Too Long')
    exit()
  s += C * L

print(s if len(s) <= 100 else 'Too Long')


'''
問題B_413_cat2・・・全探索（文字列操作）・・・解けた
時間計算量：O(N^2)
難易度51

i != j場合、文字列連携する=文字列が違う時だけ！
重複した文字列はカウントしない
'''

N = int(input())
S = [input() for _ in range(N)]
ans = set()
for s in S:
  for t in S:
    if s != t:
      ans.add(s+t)
print(len(ans))


'''
問題C_413_Large Queue・・・解けない（理解はできた）
難易度239

deque = 両橋の要素の追加、削除を高速に行える
popleft()・・・先頭の要素を取りだし削除
append()・・・末尾に要素を追加

同じ数は[要素、個数]でデータを管理する

takeで要素数を管理
'''

from collections import deque
Q = int(input())
d = deque()

for i in range(Q):
  query = list(map(int, input().split()))

  if query[0] == 1:
    x, c = query[2], query[1]
    d.append([x, c])
  else:
    k = query[1]
    total = 0
    while k > 0:
      x,c = d[0]
      take = min(k, c)
      total += x * take
      k -= take
      if take == c:
        d.popleft()
      else:
        # 先頭の要素数から取り出した現在の要素数を管理
        d[0][1] -= take
    print(total)


'''
問題C_410_Rotatable Array・・・理解はできた

難易度230

今どこのポジションを見ているかをposで管理
k ≦ 10^9
'''

N,Q = map(int, input().split())
A = list(range(1, N+1))
pos = 0
for i in range(Q):
  query = input().split()
  type = int(query[0])
  if type == 1:
    p, x = int(query[1])-1, int(query[2])
    A[(pos + p) % N] = x
  elif type == 2:
    p = int(query[1])-1
    print(A[(pos + p) % N])
  else:
    k = int(query[1])
    pos = (pos + k)%N 

'''
問題C_409_Not All Covered・・・いもす法
難易度421
'''


'''
問題B_412_Precondition・・・解けない（もう少しで解けた）
問題を解ききれなかった
プログラムの動きのイメージができていない？？

条件の整理が必要
'''

S = input()
T = input()
ans = 'Yes'
for i in range(1, len(S)):
  if S[i].isupper() and S[i-1] not in T:
    ans = 'No'
    break
print(ans)




'''
問題B_399_Ranking with Ties(問題の言い換え)・・・解けた
難易度41
時間計算量：O(N^2)
方針：大小の比較をし前の方が小さい時のみ
ランキングを+1加算していく

4
3 12 9 9

'''

N = int(input())
P = list(map(int, input().split()))

for p in P:
  rank = 1
  for j in range(N):
    if p < P[j]:
      rank += 1
  print(rank)

'''
問題B_398_Full House 3（組合せ）・・・解けた
難易度46
時間計算量：O(N）

異なるx,yのカード3枚、2枚が存在するかを判定する
場合分け→2枚を足して5になる組み合わせ存在するか？
7枚の内
3枚（x）＋２枚（y）
(3, 2), (2, 3)

Yesの時：
[3,2,1,1]
[4,2,1]
[5,1,1]
[5,1,1]

Noの時：
全部重複
[3,2,2]


カードの分布を取る
'''

A = list(map(int, input().split()))
C = set(A)
cnt = [A.count(c) for c in C]
if len(cnt) <= 1:
  print('No')
  exit()
cnt.sort(reverse=True)
print('Yes' if cnt[0] >= 3 and cnt[1] >= 2  else 'No')

'''
問題B_397：Ticket Gate Log（文字列操作）・・・解けない（ほぼ解けていた）
難易度63
ioからなる偶数の文字列を復元するための文字の挿入回数を求める

方針：
'i'の後は必ず'o'で終わるようにする
文字列連結していき
iはo, oはiとターゲットを入れ替える
'''

S = input()
ans, target = 0, 'i'
for t in S:
  # i,oの切り替え
  if target == t:
    target = ('o' if target == 'i' else 'i')
  else:
    ans += 1

if target == 'o':
  ans += 1
print(ans)


'''
問題B_395_Make Target（グリッド）・・・解けた
黒白交互のN＊Nのグリッドを作成する
難易度72

時間計算量：O(N^3）
N - 1 - iでjの範囲を調整している
'''

N = int(input())
grid = [['.' for _  in range(N)] for _ in range(N)]
for i in range(N):
  j = N - 1 - i
  if i <= j:
    for l in range(i, j+1):
      for m in range(i, j+1):
        if i % 2 == 0:
          grid[l][m] = '#'
        else:
          grid[l][m] = '.'
  print(''.join(grid[i]))

'''
問題B_391_Seek Grid（グリッド）・・・解けない（実装はできた）
難易度110

＊問題文を正確に理解し条件の整理が必要

時間計算量：O(N^4)
S[a+i]  == T[i][j]

Sの中からTと一致する座標(a,b)を探す
N-m+1通りの全探索しS[a+i][b+J] != T[i][j]
と一致しなれば、その時点でフラグをFalseへ変更する
'''

N,M = map(int, input().split())
S = [list(input()) for _ in range(N)]
T = [list(input()) for _ in range(M)]
for a in range(N-M+1):
  for b in range(N-M+1):
    ok = True
    for i in range(M):
      for j in range(M):
        if S[a+i][b+j] != T[i][j]:
          ok = False
    if ok:
      print(a+1, b+1)

'''
問題B_386：Calculator（文字列操作）・・・解けた
'00'を’X’に変換し1文字にし文字列長を出力
'''

S = input()
print(len(S.replace('00', 'X')))

'''
問題A_383：Humidifier 1（シミュレーション）・・・解けた

難易度19

必要な情報の整理
現在の水の量

方針：
入力を最初にリストで確保し最初の水だけ先に加算
その後1..N回ループし、
現在の水の量 - (次の時刻 - 前の時刻)の計算結果へ更新する
現在の水の量が0以下の場合は、maxで0にする

＊最終的に残っている加湿器の水の量を求める
'''

N = int(input())
water = 0
T = [list(map(int, input().split())) for _ in range(N)]
water += T[0][1] # 最初に水を入れる
for i in range(1, N):
  water = max(0, water - (T[i][0] - T[i-1][0]))
  water += T[i][1]
print(water)


# 別解
N = int(input())
pre, water = 0, 0
for i in range(N):
  T, V = map(int, input().split())
  water = max(0, water - (T - pre))
  water += V
  pre = T
print(water)

'''
問題B_381_1122 String（文字列操作）・・・解けた

難易度52
'''

S = input()
N = len(S)
if N % 2 != 0:
  print('No')
  exit()

C = set()
for i in range(N // 2):
  if S[2*i] != S[2*i+1] or S[2*i] in C:
    print('No')
    exit()
  else:
    C.add(S[2*i])
print('Yes')



'''
問題B_373_1D Keyboard・・・解けた
難易度54

A-Zのキーを押すまでの指の移動距離の最小値を求める

方針：
A-Zの文字列リストを作り入力された各文字のインデックス番号を検索しリストへ格納
ループの中でi+1-iの絶対値を加算していく
'''

S = input()
Key = [chr(ord('A')+i) for i in range(26)]
key = [S.find(k) for k in Key]
ans = 0
for i in range(len(S)-1):
  ans += abs(key[i+1] - key[i])
print(ans)


'''
問題B_372：3^A・・・解けない（総和）・・・貪欲法（大きい値から考える）
難易度131
3^Aiを足し合わせた結果がMになる数列との個数を求める

考察が必要
数列（0-10のいずれかの組合せ）
'''
M = int(input())
ans = []
while M > 0:
  a = 0
  while 3**(a + 1) <= M:
    a += 1
  ans.append(a)
  M -= 3**a
print(len(ans))
print(*ans)

# inadai解法
while 3**(A + 1) < M:
  A += 1

while M > 0:
  if M - 3**A >= 0:
    M -= 3**A
    ans.append(A)
  else:
    A -= 1
print(len(ans))
print(*ans)
    


'''
問題A_371_Jiro（場合分け）

余計なパターンを書かないためにまず場合分けを行う
'''

AB,AC,BC = input().split()
if AB != AC:
  print('A')
elif AC == BC:
  print('B')
else:
  print('C')

'''
問題B371_Taro

難易度21
'''

N,M = map(int, input().split())
C = set()
for i in range(M):
  A,B = input().split()
  if B == 'M' and A not in C:
    print('Yes')
    C.add(A)
  else:
    print('No')

'''
問題A_369：369（等差数列）
等差数列になる組合せを考え何通りの等差数列が作れるか求める
A,B,x
A＝ 5 B = 7

並べ方
x,A,B：
A - x = B - A
①x = 2A - B

B,x,A（分数になる）
A - x = x - B
2x = A + B
両辺を2で割る
②x = A + B / 2

x = A + B % 2 == 0を等差数列とする

A,B,x：
x - B = B - A
③x = 2B - A

AとBが等しいなら等差は１
A＋Bが割り切れたら等差であるため、3
割り切れなかったら3-1の２
'''

A,B = map(int, input().split())
if A == B:
  print(1)
elif (A + B) % 2 == 0:
  print(3)
else:
  print(2)




'''
問題B369_Piano 3・・・解けた（難しい）
難易度62

演奏を終了して時点での疲労度の最小値を求める

今右と左の手がある位置
L,R
左手と右手の操作で場合分け

'''

N = int(input())
L,R = 0, 0
ans = 0
for i in range(N):
  A, S = input().split()
  A = int(A)
  if S == 'L' and A != L:
      if L != 0:
        ans += abs(A - L)
      L = A
  elif S == 'R' and A != R:
      if R != 0:
        ans += abs(A - R)
      R = A
print(ans)


'''
問題A367_Shout Everyday（場合分け）・・・解けない（場合分けは考えることができた）
難易度43

1日は24時間
日をまたぐか
   またがないか

叫べるケース
B < C < A
A < B < C

叫ばないケース
B < A < C
'''

A,B,C = map(int, input().split())
# 日をまたいでいない場合
if B < C:
  if B < A < C:
    print('Yes')
  else:
    print('No')
# 日をまたいだ場合
else:
  if C < A < B:
    print('Yes')
  else:
    print('No')

'''
問題B367_Cut .0・・・解けた

難易度43

末尾が’０’の間繰り返す
ループ終了時の末尾が'.'の場合は'.'を削除
'''

X = list(input())
while X[-1] == '0':
  X.pop()
if X[-1] == '.':
  X.pop()

print(''.join(X))

'''
問題A364_Glutton Takahashi
難易度29

sweetが隣り合っていたらその時点でNo
N-2回繰り返す(最後がsweetの場合は気にする必要なし)
'''

N = int (input())
S = [input() for _ in range(N)]
for i in range(N - 2):
  if S[i] == S[i+1] == 'sweet':
    print('No')
    exit()
print('Yes')

'''
問題B364_Grid Walk（グリッド＆シミュレーション）・・・解けた
難易度79
時間計算量：O(N)

グリッドを上下右左に移動
マスの存在チェックおよび空きマス’.’のチェックを実装
'''

H, W = map(int, input().split())
Si,Sj = map(int, input().split())
C = [list(input()) for _ in range(H)]
X = input()
Si -= 1; Sj -= 1
for move in X:
  if move == 'L' and Sj > 0 and C[Si][Sj-1] == '.':
    Sj -= 1
  if move == 'R' and Sj < W-1 and C[Si][Sj+1] == '.':
    Sj += 1
  if move == 'U' and Si > 0 and C[Si-1][Sj] == '.':
    Si -= 1
  if move == 'D' and Si < H-1 and C[Si+1][Sj] == '.':
    Si += 1
print(Si+1, Sj+1)

'''
問題B360_Vertical Reading・・・解けない（全探索）要復習
難易度78

時間計算量：O(N^3）
制約と条件を理解し適切なrange()の範囲を実装しなければ正解できない
Sを先頭から順にw文字ごとに区切ったとき、
長さがc以上の文字列のc文字目を順番に連結した文字列がTと一致するか判定する

w文字ごとに区切る（外側）1..N
Sを先頭から順に見ていく 
長さがc以上の文字列（1..N）を順番に連結した文字列（内側）wの範囲
cはw以下になる
文字列長回文字列の連結をする（内側）

'''

# inadai解法
S, T = input().split()
N = len(S)
for w in range(1,N):
  for c in range(w):
    now = ''
    for m in range(N):
      if m % w == c:
        now += S[m]
    if now == T:
      print('Yes')
      exit()
print('No')

'''
問題B358_Ticket Counter（シミュレーション）・・・解けた
難易度43

一個前の値の保持の仕方と正確な情報の整理が課題
'''

N, A = map(int, input().split())
T = list(map(int, input().split()))
pre = 0
# 三項演算子ver
for t in T:
  pre = (A+t+(pre-t) if t < pre else A+t)
  print(pre)

# 標準
for t in T:
  if t < pre:
    pre = A+t+(pre-t)
  else:
    pre = A+t
  print(pre)

'''
問題A357_Sanitize Hands・・・解けた
Mを直接減らしていく
無駄な場合分けをしない
'''

N,M = map(int, input().split())
H = list(map(int, input().split()))
ans = 0
for h in H:
  if M < h:
    break
  M -= h
  ans += 1
print(ans)

'''
問題A356_Subsegment Reverse・・・解けた

配列の範囲指定の操作
スライスを使って実装
数列の作成
逆順 [::-1]
'''

N,L,R = map(int, input().split())
A = list(range(1, N+1)) # 数列の作成
A[L-1:R] = A[L-1:R][::-1]
print(*A)

'''
問題B356_Nutrients・・・解けない

縦の合計を求める
２次元配列の縦同士を加算する
i,jを入れ替える
iが内側、jが外側で変化していく
'''

N,M = map(int, input().split())
A = list(map(int, input().split()))
X = [list(map(int, input().split())) for _ in range(N)]
for j in range(M):
  s = 0
  for i in range(N):
    s += X[i][j]
  if s < A[j]:
    print('No')
    exit()
print('Yes')

# 355-349のB問題解く

'''
問題B_355_Piano2・・・解けた（全探索）
難易度105

時間計算量：O(N^2）だがO(N）で解ける
'''

N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
C = A + B
C.sort(); A.sort()
# for i in range(N-1):
#   for j in range(N+M):
#     if A[i:i+2] == C[j:j+2]:
#       print('Yes')
#       exit()
# print('No')

# 効率的なアルゴリズム
for i in range(N+M-1):
  if C[i] in A and C[i+1] in A:
    print('Yes')
    exit()
print('No')

'''
問題B_353_AtCoder Amusement Park（シミュレーション）・・・解けない（もう少しで正解できた）
難易度51

アトラクションをスタートさせるタイミング
K＜人数の時

難しい
'''

N, K = map(int, input().split())
A = list(map(int, input().split()))
seat = K; ans = 1
for num in A:
  if seat < num:
    seat = K
    ans += 1
  seat -= num
print(ans)

'''
問題B_352_Typing・・・解けた（シミュレーション）
難易度55

現在打った入力位置をsiで保持させる
si = 今Sの何文字目を打とうとしているかを保持する変数
時間計算量の工夫が必要
'''
S = input()
T = input()
ans = []; si = 0
for i in range(len(T)):
  if S[si] == T[i]:
    ans.append(i+1)
    si += 1
print(*ans)

'''
問題A_352_Zero Sum Game・・・解けた

合計が0になるのは合計をマイナスした時だけ
ーー＝＋
＋＝ー
'''

N = int(input())
A = list(map(int, input().split()))
print(-sum(A))

'''
問題B_349_Commencement・・・解けない（もう少しで解ける）

各文字列の登場回数の頻度分布を取る
カウントの頻度分布を取る
'''
S = input()
C = set(S) # 重複を除く
l = [S.count(s) for s in C] 
cnt = [l.count(i) for i in l]
print('Yes' if all(c == 2 for c in cnt) else 'No')

'''
問題B348_Farthest Point（最大の距離・座標問題）・・・解けない
難易度79

ユークリッド距離を求める
(x1 - x2)**2 + (y1 - y2)**2
ルートは浮動小数になる場合があり、計算に誤差が出るため。
2乗した整数を比較する

i≠Jの時のみ、それぞれの距離の最大値の計算を行う
最大値の距離が前の距離より大きい時のみ最大値とindexを更新
'''

def sq(x: int) -> int:
  return x * x

N = int(input())
# x,yの座標リストを作成
X, Y = [0]*N, [0]*N
for i in range(N):
  X[i], Y[i] = map(int, input().split())

for i in range(N):
  d, idx = 0, -1
  for j in range(N):
    if i == j:
      continue
    # 距離を求める
    d_max = sq(X[i]-X[j]) + sq(Y[i]-Y[j])
    # 距離が大きい距離に更新された時だけ距離を更新する
    if d_max > d:
      d, idx = d_max, j
  print(idx + 1)

'''
問題B_347_SubString・・・集合（空集合の性質の問題）&全探索・・・解けた
時間計算量：O(N^2)

難易度81
yayの場合 = ５個
y,ya,yay
a,ay
'''

S = input()
C = set(); N = len(S)
for i in range(N):
  for j in range(i, N):
    if S[i:j+1] not in C:
      C.add(S[i:j+1])
print(len(C))

'''
問題B_346_Piano・・・解けない（全探索）
'''

'''
問題B_345_Integer Division Returns（切り上げ除算）・・・解けた
難易度91
切り上げ除算の求め方
a + b - 1 / b

a = X
b = 10

x + 10 - 1 / 10 = x + 9/10
'''
X = int(input())
print((X + 9) // 10)

'''
問題B_341_Foreign Exchange・・・解けない
通貨の両替の問題？？

計算の仕方がわからない
問題の意味：
国Nの通貨を最大でどれくらい大きく両替できますか？

N- 1回Ai＋1にAi//Si＊Tiを行い最終的なA[N]が求めたい答えになる
'''

# 他の人の解法
N = int(input())
A = list(map(int, input().split()))
for i in range(N-1):
  S,T = map(int, input().split())
  A[i+1] += A[i] // S * T
print(A[-1])



