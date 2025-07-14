from collections import deque

'''
問題C_414_Palindromic in Both Bases （回文）・・・解けない（要復習）

A進法の回文の総和を求める

回文とは
前から読んでも後ろから読んでも同じになる文字（言葉）のこと

回文の性質：
1. 左右対称になる
2. 文字数が偶奇でも回文になる
3. 真ん中の文字はなんでもいい（対応する相手が自分しかいないから）

文字をひっくり返して元の文字（または数字）と同じか比較する
'''




'''
問題C_413：Large Queue
時間計算量：O(Q)

データ構造はdequeを使う
deque・・・両橋に対して要素の追加や削除が高速で行える
deque()・・空のdequeを作成

追加
append(要素)・・・末尾に要素を追加
appendleft(要素)・・・先頭に要素を追加

削除
pop()・・・末尾の要素を削除しその値を取得
popleft()・・・先頭の要素を削除してその値を取得

拡張
extend([要素])・・・末尾に複数の要素を追加
extendleft([要素])・・・先頭に複数の要素を追加

Q回のクエリを処理
空のdequeを作成
xをc個dequeへの末尾へペアでで管理・・・([3, 2]) →[3, 3]と同じ
([x, c])・・・例）3を2を追加のイメージ

k個の要素を削除
要素の個数をtakeで管理する

要素の取り出しはx * take
取り出した要素数＝take
都度、k - takeをし要素数を管理
取り出した要素数はcと一致した場合は、先頭の要素を削除
まだ取り出し終わっていない場合は、先頭の要素の個数-takeする

以上をk回繰り返す
'''

Q = int(input())
d = deque()

for _ in range(Q):
  query = input().split()
  # 追加
  if query[0] == '1':
    x, c = int(query[2]), int(query[1])
    d.append([x, c])
  # 削除
  else:
    k = int(query[1])
    total = 0
    while k > 0:
      # 削除する要素を取り出す
      x ,c = d[0]
      take = min(k ,c) # 要素の個数を管理
      total += x * take
      k -= take
      # 要素を全て取り出し終わった場合
      if take == c:
        d.popleft()
      else:
        d[0][1] -= take
    print(total)



'''
問題C_410_Rotatable Array・・・理解はできた
計算量の工夫を問われる問題
難易度230

タイプ3のk回先頭の要素をpopし末尾に移動する操作をどう工夫するか？？

今どこのポジションを見ているかをposで管理
k ≦ 10^9

(pos + 入力値) % N でNとイコールの時0に戻る
＊整数の循環をしているため配列要素外にアクセスしない
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
問題C_408_Not All Covered・・・いもす法（累積和のアルゴリズム）
難易度223
いもす法（imos）を使う

1. [0]*(N+1)のimosリストを準備
2. L, R区間の開始（+1）と終了地点（-1）をimosへ記録していく
  配列の添え字は0開始のため、L - 1しておく
3. 累積和の計算
   imos[i+1] += imos[i]をN回繰り返す 
4. 累積和の結果から最小値を取り出す（求める答え）


求めること：
砲台に守られていない城壁が存在する状態にするために、
破壊する必要がある砲台の個数の最小値

N = 壁の数
M = 砲台の数
L, R = 砲台を守っている区間
L ≦ R（区間）

砲台が守っているのは
N - M + 1 = 7(6通り)
10 4
砲台1：1 6・・・1..6
砲台2：4 5・・・4..5
砲台3：5 10・・・5..10
砲台4：7 10・・・7..10

'''
N, M = map(int, input().split())
imos = [0] * (N+1) # 累積和を取るためのリスト

for i in range(M):
  L,R = map(int, input().split())
  L -= 1
  # 砲台の守っている区間の開始地点と終了地点を記録する  
  imos[L] += 1
  imos[R] -= 1

# 累積和を取る（累積和の結果末尾（最終的な計算結果が0になる））
for i in range(N):
  imos[i+1] += imos[i]

ans = 1e9 # 10^9
# 最小値を求める
for i in range(N): # N+1の配列に対してN回しか回らないので1回少ない
  ans = min(ans, imos[i])
print(ans)

'''
問題C_407_Security 2
'''




'''
問題028_Frog・・・動的計画法
＊漸化式の概念の理解が必要

カエルが支払うコストの最小値を求める

dp表を作り前の計算結果を記録する
次の計算時に前の計算結果を参照し計算の重複をなくす
最終的な総和の計算結果のN番目が求めるコストの総和


入力例）
4
10 30 40 20

1通り
足場1-足場1のコスト：0
足場1-足場2のコスト：0 + |30-10| = 0 + 20 = 20

[0, 20, 0, 0]

2通り
足場3への行き方
足場1-足場3：0 + |40-10| = 0 + 30 = 30
足場2-足場3：20 + |40-30| = 20 * 10 = 30


[0, 20, 30, 0]
足場4への行き方
足場2-足場4：20 + |20-30| = 30
足場3-足場4：30 + |20-40| = 50

[0, 20, 30, 30]

答え：３０
dpの結果：[0, 20, 30, 30]
dp[N-1]が答え
'''

N = int(input())
H = list(map(int, input().split()))
dp = [0]*N # 計算結果を保持するN個要素を持つdp表を作成
for i in range(1, N):
  if i == 1:
    dp[i] = dp[i-1] + abs(H[i] - H[i-1])
    continue
  dp[i] = min(dp[i-2] + abs(H[i] - H[i-2]),
              dp[i-1] + abs(H[i] - H[i-1]))
print(dp[N-1])

'''
問題029_Climb Stairs・・・動的計画法
＊N段の階段を登るときに、N弾目にたどり着くまでの移動方法が何通りあるか求める
最後のN段目を１段で上がる：N - 1
最後のN段目を２段で上がる：N - 2

求める答え
dp[N] = dp[N-1] + dp[N-2]

1段目から開始
階段0-1段：1

前の項と次の項を足していくのを繰り返す（漸化式）
'''

N = int(input())
dp = [0] * (N+1)
dp[0], dp[1] = 1, 1

for i in range(2, N+1):
  dp[i] = dp[i-1] + dp[i-2]
print(dp[N])






