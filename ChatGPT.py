'''
苦手克服
- シミュレーション：変数の状態管理, min/maxの考え方
- 全探索（線形探索、O(N^2）通り、組合せ全探索（ビット全探索）、全列挙）
- 累積和
'''



'''
シミュレーション問題の練習
'''
# LV2
L,X,N = map(int, input().split())
juice = 0
for _ in range(N):
  juice += min(X, L- juice)
print(juice)

# LV2
E,X,N = map(int, input().split())
S = input()
energy, ans = E, 0
for s in S:
  if s == 'x':
    if energy < X:
      break
    energy -= X
  ans += 1
print(ans)


# LV３
E,N = map(int, input().split())
S = input()
W = list(map(int, input().split()))
R = list(map(int, input().split()))
SS = list(map(int, input().split()))
energy = 0  
# maxは負の値にしないため, max(0, e - 消費量)
# minは上限を超えないため, min(E, e + 回復量)
for i, s in enumerate(S):
  if s == 'w':
    energy = max(0, energy - W[i])
  elif s == 'r':
    energy = min(E, energy + R[i])
  else:
    energy = max(0, energy - SS[i])
print(energy)


# LV3
W,N = map(int, input().split())
S = input()
U = list(map(int, input().split()))
F = list(map(int, input().split()))
water = 0
for i, s in enumerate(S):
  if s == 'w':
    water = max(0, water- U[i])
  if s == 'f':
    water = min(W, water + F[i])
print(water)

'''
全探索の練習問題
線形探索、全探索（組合せ、全列挙、ビット全探索）
'''

# LV1 O(N)
N = int(input())
for i in range(1, N+1):
  if i % 3 == 0:
    print(i)

# LV2 O(N^2)
N = int(input())
M = int(input())
for a in range(1, N+1):
  for b in range(1, N+1):
    if a == b or b < a: continue
    if a + b == M:
      print(a, b)

# LV3 O(N^3)
'''
a + b + c = M
c = M - a - b
2つの値が分かればcの値は確定できる
'''


# LV3 O(N^3)→O(N^2）へ修正 c = M - a - b
# print(1 << i) # 1, 2, 4, 8, 16, 32, 64, 128（2進数の桁）
N = int(input())
M = int(input())
for a in range(1, N+1):
  for b in range(1, N+1):
    c = M - a - b
    if a == b or a == c or b == c: continue
    total = a + b + c
    if a < b < c and total == M:
      print(a, b, c)


# LV4・・・組合せ全探索（bit全探索：難しい）
N = int(input())
M = int(input())
A = list(map(int, input().split()))
found = False

for bit in range(1 << N): # 2^N通り試す
  total = 0
  for i in range(N):

    # 1をi回左にズラす（1が立っている場所i個ずつ移動させる）
    # i番目を選んでいるかをチェック
    # bit演算
    if bit & (1 << i):
      total += A[i]
  
  if total == M:
    found = True
    break
print('Yes' if found else 'No')

# LV4 時間計算量：O(2N*N)
N,X = map(int, input().split())
A = list(map(int, input().split()))
total, ans = 0, 0
for bit in range(1 << N):
  total = 0
  for i in range(N):
    if bit & (1 << i):
      total += A[i]
      
  if total == X:
    ans += 1
print(ans)

# LV4
N,X = map(int, input().split())
A = list(map(int, input().split()))
ans = -1
for bit in range(1 << N):
  total = 0
  for i in range(N):
    if bit & (1 << i):
      total += A[i]
  if total <= X:
    ans = max(ans, total)

print(ans) 

# LV4
N,X = map(int, input().split())
A = list(map(int, input().split()))
found = False
for bit in range(1 << N):
  total = 0
  for i in range(N):
    if bit & (1 << i):
      total += A[i]
  if total == X:
    found = True
    break
print('Yes' if found else 'No')


N,X = map(int, input().split())
A = list(map(int, input().split()))
ans = 0
for bit in range(1 << N):
  total = 0
  for i in range(N):
    if bit & (1 << i):
      total += A[i]
  if total == X:
    ans += 1
print(ans)

N,X = map(int, input().split())
A = list(map(int, input().split()))
INF = float('inf')
ans, cnt = INF, 0

for bit in range(1 << N):
  total,cnt = 0,0
  for i in range(N):
    if bit & (1 << i):
      total += A[i]
      cnt += 1
  if total == X:
    ans = min(ans, cnt)
print(ans if ans != INF else -1)


N,X = map(int, input().split())
A = list(map(int, input().split()))
for bit in range(1 << N):
  total = []
  for i in range(N):
    if bit & (1 << i):
      total.append(A[i])
  if sum(total) == X:
    print(*total)

N,X = map(int, input().split())
A = list(map(int, input().split()))
INF = float('inf')
cnt, ans = INF, []
for bit in range(1 << N):
  total = []
  for i in range(N):
    if bit & (1 << i):
      total.append(A[i])
  if sum(total) == X:
    if len(total) < cnt:
      cnt = len(total)
      ans = [total]
    elif len(total) == cnt:
      ans.append(total)

for a in ans:
  if len(a) == cnt:
    print(*a)


# LV4 重複しない合計＝Xになる組合せの件数を求める
# setを使う際は順番を保持しない為、昇順にする
N,X = map(int, input().split())
A = list(map(int, input().split()))
unique_sets = set()
for bit in range(1 << N):
  total= []
  for i in range(N):
    if bit & (1 << i):
      total.append(A[i])
  if sum(total) == X:
    unique_sets.add(tuple(sorted(total)))
print(len(unique_sets))


# LV4 合計＝Xになる重複を除いた最小の要素数の組合せを求める
N,X = map(int, input().split())
A = list(map(int, input().split()))
unique_sets = set()
min_len = float('inf')

for bit in range(1 << N):
  total = []
  for i in range(N):
    if bit & (1 << i):
      total.append(A[i])
  
  if sum(total) == X:
    # 重複した組合せを追加しない為、昇順にする
    sorted_total = tuple(sorted(total))
    if len(total) < min_len:
      unique_sets = {sorted_total} # リセット
      min_len = len(total)
    elif len(total) == min_len:
      unique_sets.add(sorted_total)

# 最小値の組合せを出力
for ans in unique_sets:
  print(list(ans))

'''
累積和
'''

# LV1 O(N+Q)
N,Q = map(int, input().split())
A = list(map(int, input().split()))

# 累積和の前処理(0-indexed)
S = [0] * (N + 1)
for i in range(N):
  S[i + 1] = S[i] + A[i]

# Q個のクエリを処理
for _ in range(Q):
  l, r = map(int, input().split())
  print(S[r+1] - S[l])

# LV2 l〜r区間の偶数の要素数を求める
N,Q = map(int, input().split())
A = list(map(int, input().split()))
# 累積和の前処理 偶数：1 奇数：0
sum_flg = [1 if a % 2 == 0 else 0 for a in A]

# 累積和の前処理
S = [0] * (N + 1)
for i in range(N):
  S[i + 1] = S[i] + sum_flg[i]

for _ in range(Q):
  l,r = map(int, input().split())
  print(S[r + 1] - S[l])


N,Q = map(int, input().split())
A = list(map(int, input().split()))
sum_flg = [1 if a % 2 != 0 else 0 for a in A]

# 累積和の前処理
S = [0] * (N + 1)
for i in range(N):
  S[i + 1] = S[i] + sum_flg[i]

# Q個のクエリを処理
for _ in range(Q):
  l,r = map(int, input().split())
  print(S[r + 1] - S[l])

# [l, r] = r - l + 1 l以上r以下 例） l = 2, r = 4 [2, 3, 4] = 要素数＝３ 4 -2 + 1 = 3（区間：両端含む）
# [l, r) = r - l l以上r未満
N,Q = map(int, input().split())
A = list(map(int, input().split()))
even_flg = [1 if a % 2 == 0 else 0 for a in A]
odd_flg = [1 if a % 2 != 0 else 0 for a in A]

S = [0] * (N + 1)
S1 = [0] * (N + 1)
for i in range(N):
  S[i + 1] = S[i] + even_flg[i]
  S1[i + 1] = S1[i] + odd_flg[i]

for _ in range(Q):
  l,r = map(int, input().split())
  even = S[r + 1] - S[l]
  odd = S1[r + 1] - S1[l]
  print('Yes' if even == odd else 'No')
  

# LV3 区間の中で偶数より多い奇数の区間の個数の最大値を数える
# 区間の長さ＝[l, r] = l 以上、r以下、r - l + 1
# 奇数の数＝(r - l + 1) - 偶数の数
N = int(input())
A = list(map(int, input().split()))
even_flg = [1 if a % 2  == 0 else 0 for a in A]
even_sum = [0] * (N + 1)

for i in range(N):
  even_sum[i + 1] = even_sum[i] + even_flg[i]
  

ans = 0
for l in range(N):
  for r in range(l, N+1):
    even = even_sum[r + 1] - even_sum[l]
    odd = (r - l + 1) - even
    if odd > even:
      ans += 1
print(ans)


# LV4 区間[l, r]の偶数と奇数の個数が等しい数を求める
N = int(input())
A = list(map(int, input().split()))
EVEN = [1 if a % 2 == 0 else 0 for a in A]
S = [0] * (N + 1)
# 偶数の累積和の前処理
for i in range(N):
  S[i + 1] = S[i] + EVEN[i]

ans = 0
for l in range(N):
  for r in range(l, N):
    even = S[r + 1] - S[l]
    odd = (r - l + 1) - even
    if even == odd:
      ans += 1
print(ans)

# 2次元累積和 LV5
H,W = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(H)]
Q = int(input())

# 偶数の個数を数える累積和用のリスト 偶数：1 奇数；0
EVEN = [[1 if a % 2 == 0 else 0 for a in row] for row in A]
# 2次元累積和の初期化
S = [[0] * (W + 1) * H for _ in range(H + 1)]

print(A)
print(EVEN)
print(S)

# 偶数の個数の累積和を準備
for i in range(H):
  for j in range(W):
    S[i + 1][j + 1] = (
      S[i + 1][j] + S[i][j + 1] - S[i][j] + EVEN[i][j]
    )

for i in range(Q):
  r1,c1,r2,c2 = map(int, input().split())
  # ここの考え方がわからない
  ans = (
    S[r2 + 1][c2 + 1]
    - S[r1][c2 + 1]
    - S[r2 + 1][c1]
    + S[r1][c1]
  )
  print(ans)

  
  '''
  min(), max()の理解
  '''

# LV1
a,b = map(int, input().split())
print(min(a, b), max(a, b))

# LV2
def f(x: int) -> int:
  return max(0, x)

x = int(input())
print(f(x))

# LV3
N = int(input())
A = list(map(int, input().split()))
print(f'min: {min(A)}')
print(f'max: {max(A)}')

# 別解
min_val = 1e6
max_val = -1

# A[i] < min_valの場合のみ最小値を更新
for i in range(N):
  min_val = min(min_val, A[i])

# max_val < A[i]の場合のみ最大値を更新
for i in range(N):
  max_val = max(max_val, A[i])

print(f'min: {min_val}')
print(f'max: {max_val}')

# LV4 範囲制限（クリッピング）
# xがL以上R以下に収める
x,L,R = map(int, input().split())
print(min(max(x, L), R))

# LV5 累積
N = int(input())
A = list(map(int, input().split()))
min_val = 1e9
ans = []
for i in range(N):
  min_val = min(min_val, A[i])
  ans.append(min_val)
print(*ans)

# LV６ 範囲の幅を求める
N = int(input())
A = list(map(int, input().split()))
print(max(A) - min(A))

# 別解
min_val = 1e6
max_val = -1

for i in range(N):
  min_val = min(min_val, A[i])

for i in range(N):
  max_val = max(max_val, A[i])
print(max_val - min_val)

# LV7(ジェネレータ式)
N = int(input())
A = list(map(int, input().split()))
print(min(A[i] for i in range(N) if A[i] > 0))

# LV8
N = int(input())
A = list(map(int, input().split()))
print(max(abs(A[i+1] - A[i]) for i in range(N-1)))
