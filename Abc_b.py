'''
問題B_042：文字列大好きないろはちゃんイージー・・・解けた
時間計算量：O(N^2)

昇順に並べ替える
文字数繰り返し文字列連結して結果を返す
'''
def iroha_b(l: list) -> list:
  l.sort()
  result = ''
  for s in l:
    result += s
  return result

print(iroha_b(['dxx', 'axx', 'cxx']))

'''
3. 問題B_081：Shift Only（200点）・・・解けない
時間計算量：O(N)

＊偶奇のチェックと性質の理解
黒板にN個の正の整数A1〜Anが書かれている。
すぬけ君は黒板に書かれている整数がすべて偶数である時、次の操作を行う
・黒板に書かれている正数すべてを2で割ったものに置き換える

すぬけ君は最大で何回操作を行うことができるか
'''
x = [16, 12, 24]
y = [4, 4, 2]
def shift_only(n: int, numbers: list) -> int:
  # 割り切れなくなった時に処理終了
  count = 0
  # 無限ループ
  while True:
    # 奇数が1つでもあれば終了（ジェネレーター式：Trueが一つでもあれば結果が確定）
    if any(num % 2 != 0 for num in numbers):
      return count
    # すべての数を2で割る
    numbers = [num // 2 for num in numbers]
    count += 1

# print(shift_only(3, x))
# print(shift_only(3, y))

'''
4. 問題B_087：Coins（200点）・・・解けない
時間計算量：O(N^3)

＊全探索の問題

500円玉：A枚
100円玉：B枚
50円玉：C枚

これらの中から何枚か選び合計金額をちょうどX円にする
方法は何通りか。

制約
・0 ≦ A,B,C ≦ 50
・A + B + C ≧ 1
・50 ≦ X ≦ 20000
・A,B,Cは正数
・Xは50の倍数

組合せ
1. A(500 * 0) + B(100 * 1) + C(50 * 0) = X(100)
2. A(500 * 0) + B(100 * 0) + C(50 * 2) = X(100)

A + B + C = X
合計金額が一致したらカウントする

枚数が+1枚ずつ加算していく

(A + 1)(B + 1)(C + 1)通り
'''
def coins(a :int, b: int, c: int, x: int) -> int:
  # ABCのコインを//50で割りコイン1枚の金額を出す
  count = 0
  for i in range(a + 1): # 0, 1, 2
    for j in range(b + 1): # 0, 1, 2
      for k in range(c + 1): # 0, 1, 2
        total = 500*i + 100*j + 50*k
        # 合計金額が一致したらカウント
        if total == x:
          count += 1
  return count

print(coins(2, 2, 2, 100))

'''
5. 問題B_083：SomeSums（200点）・・・解けない
時間計算量：O(N^2) / 10

＊10進法表記の考え方

1 ≦ N ≦ 10^4
1 ≦ A ≦ B ≦ 36

1以上N以下の整数のうち、10進法で各桁の和が
A以上B以下であるものについて、総和を求めてください。

0になるまで繰り返す
10で割ったあまり
10で割りiの数を0にする

四則演算の使い方・考え方
割り算（商）・あまり（余）・足し算（和）・引き算（ー）

'''

# 各桁の総和
def sum_sums(n: int, a: int, b: int) -> int:
  total= 0
  for i in range(1, n + 1): # 1-20
    sum = 0
    n = i
    # 格桁の和がA以上B以下かどうか
    while i > 0: # nが0になるまで繰り返す
      sum += i % 10
      i //= 10
    if a <= sum <= b:
      total += n
  return total

print(sum_sums(20, 2, 5))

'''
6. 問題B_088：CardGameForTwo（200点）・・・解けた

時間計算量：O(N^2)

N＝カードの枚数
ai = カードに書かれている数

制約
1 ≦ N ≦ 100
a ≦ ai ≦1 00

降順に並べ替える
交互にカードを引く処理は偶奇で判断する
'''

def card_game_for_tow(n: int, cards: list) -> int:
  cards.sort(reverse=True) # 降順に並べ替える
  a, b = 0, 0
  # AliceとBobが交互にカードを選ぶ
  for i, card in enumerate(cards):
    if i % 2 == 0: # 先頭はAlice
      a += card
    else:
      b += card  # 後攻はBob
  return a - b

print(card_game_for_tow(3, [2, 7, 4]))
print(card_game_for_tow(10, [1, 5, 9, 44, 20, 13, 17, 11, 2, 2]))

'''
7. 問題B_085：KagamiMochi（200点）・・・解けた

時間計算量：O(1)

データ構造：辞書
キー：値 / 値:index
重複したキー（値）を除く
'''

def kagami_mochi(n: int, d: list) -> int:
  k = {}
  for data in d:
    k[data] = '_'
  return len(k)

print(kagami_mochi(4, [8, 10, 8, 6]))
print(kagami_mochi(3, [15, 15, 15]))
print(kagami_mochi(7, [50, 30, 50, 100, 50, 80, 30]))

'''
問題B_042：文字列大好きいろはちゃんイージー・・・解けた
時間計算量：O(N^2)
N個の個数分文字列入力を受け取りsortで昇順に並べ替える
'''

N,L = map(int, input().split())
S = [input() for _ in range(N)]
S.sort()
print(S)

'''
問題B_043：バイナリハックイージー・・・解けない
時間計算量：O(N)

データ構造：リスト
データは末尾に追加される


0, 1, Bのパターンに分ける
B以外の場合はリストへ追加
Bが押されたら要素がある場合のみリストの末尾を削除

 入力のデータがパターン分けできるか考える
 データ構造を考える
 pop()は引数指定なしの場合リストの末尾を削除する
 append()はリストの末尾にデータを追加していく
 ''.join(コレクション)を使えばコレクション型を文字列型へ変換できる
 文字列の値はindex番号を参照し変更できない
'''
s = list(input())
result = []
for c in s:
  if c == '0':
    result.append(c)
  elif c == '1':
    result.append(c)
  elif c == 'B':
    if result: # リストが空でなければ末尾を削除
      result.pop()
print(''.join(result))

'''
問題B_044：美しい文字列・・・解けた
時間計算量：O(N^2)
＊どの英小文字もw 中に偶数回出現する。
偶 + 偶 = 偶数
奇 + 奇 = 偶数
偶 + 奇 = 奇数

Sの文字列をリストに変換し渡す
文字列長が偶数の時のみ以下の処理を繰り返す
-----------------------------------------------------
一文字ずつ順にリストの件数分存在チェックする
文字列が一時リストに存在しない時のみ
文字列カウントを検索し、文字列カウントが奇数ならNoを返し処理修了
偶数なら一時リスト文字列を追加
-----------------------------------------------------
'''
def beautiful_str(w: list) -> str:
  if len(w) % 2 != 0:
    return 'No'
  temp = [0]*2
  for s in w:
    if s not in temp:
      cnt = w.count(s)
      if cnt % 2 != 0:
        return 'No'
    temp.append(s) 
  return 'Yes'

w = list(input())
print(beautiful_str(w))

'''
問題A_045：3人でカードゲームイージー・・・解けた
時間計算量：O(1)
データ構造：辞書
キー：A,B,C
値  ：SA ,SB,SC

A,B,Cさん
Aさんのターンから始まる
各プレイヤーの動き（共通）
・現在自分のターンである人がカードを1枚上持っているなら先頭のカードを捨てる
・捨てたカードアルファベットの書かれている人のターンとなる
・現在自分のターンである人がカードを1枚も持っていないなら、その人がゲームの勝者となりゲームは終了する

データ
[a,c,a]
[a,c,c,c]
[c,a,]

[a,b,c,b]
[a,a,c,b]
[b,c,c,c]

[a], [b], [c]
[b], [a], [c]

'''

# リファクタリング後
def card_game_easy(A: list, B: list, C: list) -> str:
  i = 0; is_playing = True
  player = {'a': A, 'b': B, 'c': C}
  # Aは最初のターン
  key = 'a'
  while is_playing:
    card = player[key]
    # 現在のターンのプレイヤーがカードを1枚以上持ってる時のみ要素を削除
    if len(card) >= 1:
        card = card[i]
        player[key].pop(0)
        key = card # 次のプレイヤーに切り替える
    else:
      is_playing = False
  return key.upper()

A,B,C = [list(input()) for _ in range(3)]
print(card_game_easy(A, B, C))

'''
問題B_046：AtCoDeerくんとボール色塗り（高校数学）
時間計算量：O(N)

組合せの計算（順番を区別しない）
隣り合うボールの色は異なる色にしなければならない

1番目ボールの色・・・K通り
2番目以降のボールの色・・・K-1通り
→1番目のボールと同じ色を選べないため。
同様に3番目のボールは2番目のボールと異なる色を選ぶ必要があるので,K - 1通り


1番目のボール: ( K ) 通り
2番目から ( N ) 番目のボール:K(K-1)^N-1通り



まず、左端のボールの塗り方は K
 通り考えられる。
次に、その右隣のボールの塗り方は K−1
 通り（左端と異なる必要があるため）
次に、その右隣のボールの塗り方は K−1
 通り（左端と異なる必要があるため）
...
最後に、右端のボールの塗り方は K−1
 通り（左端と異なる必要があるため）

K×(K−1)×(K−1)×⋯×(K−1)=K(K−1)^N−1通り
→階乗の計算方法と考え方

総数
K × (K-1) ** (N-1)

＊N - 1とは先頭の1番目のボールの色（K通り）を除いた2番目以降のボールの数
(K-1)**(N-1)＝1番目のボールの色（K通り）を除いた2番目以降からN番目までの総数

(1番目の選択肢)×(2番目以降の組み合わせ)=K×(K−1)**N−1


'''
N,K = map(int, input().split())
ans = K * (K-1) ** (N-1)
print(ans)

'''
問題B_081：Shift Only・・・解けた
入力値のすべての整数が偶数であるか確認する
% 2 == 0
各要素が割りれきれるまで繰り返す
1周した時点で操作回数をカウントする
割り切れなくなった時点でカウントを返し処理終了

ジェネレータ式：
any(条件式 for n in コレクション)

リスト内包表記：
変数 = [変数へ格納する値 for x in コレクション]
'''

def shift_only_practice(N: int, A: list, count=0) -> int:
  # すべての整数を2で割ったものに置き換える
  while True:
    # 奇数が1つでもあれば終了（ジェネレーター式：Trueが一つでもあれば結果が確定）
    if any(num % 2 != 0 for num in A):
      return count
    # すべての数を2で割る
    A = [num // 2 for num in A]
    count += 1

# 最初のver
def shift_only_first_ver(N: int, A: list, count=0) -> int:
  A_copy = A.copy()
  while True:
    for i, data in enumerate(A_copy):
      if data % 2 == 0:
        A_copy[i] = data // 2
      else:
        return count
    count += 1

N,A = int(input()), map(int, input().split())
print(shift_only(N, list(A)))


'''
問題B_407：
時間計算量：O(N^2)

誤差が 10^−9以下のとき正答と判定される

# ２つのサイコロを振る 6 * 6 = 36通り
確率 = aが起こる場合の数/全てが起こりうる場合の数
p = a/n
2 つの出目の合計がX 以上である。
2 つの出目の差の絶対値がY 以上である。

補足：
組合せを確認するときに、
set((x, y))のようにタプルを追加
print(*set)で要素を表示できる
'''

x, y = map(int, input().split())
cnt = 0
for a in range(1, 7):
  for b in range(1, 7):
    if a + b >= x or abs(a - b) >= y:
      cnt += 1
print(cnt / 36)

'''
問題B_406：Product Calculator（12分）
時間計算量：O(N)
(N+1)/2 = 3回？

電卓の操作回数：
N=操作回数
K=表示できる桁数
K+1以上の場合は1を表示
電卓の結果を操作回数分、累積していく
'''
n, k = map(int, input().split())
ai = list(map(int, input().split()))
ans = 1
for i in ai:
  ans *= i
  if k + 1 <= len(str(ans)):
    ans = 1
print(ans)

'''
問題B_405：
時間計算量：O(N^2)

方針：
1以上M以下の整数が含まれていなければ処理を処理終了
含まれていた場合、リストの末尾を削除をカウントする
'''

n, m = map(int, input().split())
a_l = list(map(int, input().split()))
ai = [i for i in range(1, m+1)]
cnt = 0
for _ in range(n):
  if all(a in a_l for a in ai):
    cnt += 1
    a_l.pop()
  else:
    break
print(cnt)

'''
問題B_404：Grid Rotation・・・解けない
時間計算量：O(N^2)

S = Tとなるグリッドを操作した時の最小回数を求める

.・・・白
#・・・黒

方針：
最小値の操作回数＝回転した回数＋異なるマスの数

# 90度回転を4回(0 * 90, 1 * 90, 2 * 90, 3 * 90）試す
# 現在のSとTの異なるマスの数をカウントし異なるマスの数の最小値を更新する
# 回転時に新しいグリッドを毎回作成し現在のグリッドを返す
# 回転後の座標は(i, j)→(j, N - 1 - i)となる 新しい行 = j、新しい列 = N - 1 - i

以上のチェックを4パターン（90°回転 0,1,2,3）を試し一番最小値の値を返す
※常に最小値に更新される設計

回転：
回転により行が列に変わるとき、行の順序が反転（上→下 が 右→左）



縦＊横

i=行 j=列

4
###.
..#.
..#.
..#.
...#
...#
###.
....

'''

# 右側に90°回転後の新しいグリッドを返す
# 座標：(j, N - 1 - i)
def rotate(grid: list, N: int) -> list:
  # 白マスのN*Nの新しいグリッドを作成
  res = [['.' for i in range(N) ] for j in range(N)]
  for i in range(N):
    for j in range(N):
      res[j][N - 1 - i] = grid[i][j] # 90°回転後のグリッドを作成する
  return res


# グリッドSとTの異なるマスの数をカウント
def diff_count(n: int ,s: list, t: list) -> int:
  diff_cnt = 0
  for i in range(n):
    for j in range(n):
      if s[i][j] != t[i][j]:
        diff_cnt += 1
  return diff_cnt


# 入力
n = int(input())
# グリッドの為二次元配列(i, j)で持たせる
s = [list(input()) for _ in range(n)]
t = [list(input()) for _ in range(n)]

# メイン処理
ans = 10**9 # 暫定最小値
cur = s # 現在のグリッド
for row_count in range(4):
  # 異なるマスの数をカウントし最小値を更新する
  ans = min(ans, diff_count(n, cur, t)+row_count)
  # 右側に90°回転後の現在のグリッドを返す
  cur = rotate(cur, n)
print(ans)

'''
問題B_403：Four Hidden・・・解けない（解けない）

T は長さ4以上10以下の英小文字と ? からなる文字列
T にはちょうど 4つの ? が含まれる
Uは長さ1以上∣T∣以下の英小文字からなる文字列

方針：
場所の全探索
T - U + 1通り

T=9 U=5
9 - 5 + 1

Tが？かつT=Uがなら’Yes’→含まれている
Tが?ではなくT≠Uなら'No'→一含まれていないと言える
文字列SがUを連続部分文字列として含む位置

T = t a k ? ? a ? h ?
U = nashi
'''
t = input()
u = input()
for i in range(len(t) - len(u) + 1):
  ans = True
  for j in range(len(u)):
    if t[i + j] != '?' and t[i + j] != u[j]:
      ans = False
      break
  if ans:
    print('Yes')
    exit()
print('No')


'''
問題B_402：Restaurant Queue・・・解けた
後ろから入って前から取り出す

時間計算量：O(N)

データ構造：リスト
メニュー番号を格納

方針：

1, 2のクエリのパターンに処理を分ける

行列を管理するリストを作成

行列の人数分繰り返す
1の場合：
行列管理リストの末尾にメニュー番号を追加

2の場合：
行列管理リストの先頭の要素を取り出す
リストの先頭を削除
'''

n = int(input()) ;queue = []
for _ in range(n):
  type = list(map(int, input().split()))
  if type[0] == 1:
    queue.append(type[1])
  elif type[0] == 2:
    print(queue[0])
    queue.pop(0)

'''
問題B_401：Unauthorized
認証をエラーを受け取った回数を求める

N回の操作パターン
・login
・logout
・public
・private

初期はログインしていない状態
ログイン済みかどうかを判定

条件
ログインしてない状態に限り、
非公開ページにアクセスしたとき、ウェブサイトは認証エラー

ログインしている状態：
認証エラーにならない

方針：
private かつ ログインしていない状態の時に認証エラーをカウントする
login状態をis_loginで管理
logoutしたらis_loginをFalseへ変えログアウト状態にする
'''
n = int(input())
authentication = [input() for _ in range(n)]
cnt = 0; is_login = False
for user in authentication:
  if user == 'login':
    is_login = True
  if user == 'private' and not is_login :
    cnt += 1
  if user == 'logout':
    is_login = False
print(cnt)

'''
問題B_400：Sum of Geometric Series・・・解けた
   M
X= ∑ N^i
   i=0
初期値＝0
M=繰り返し条件、iがMの値を超えるまで繰り返し計算を行う
N^i=実際に繰り返す計算式
例）
N=7の場合
x = 1 + 7^1 + 7^2 + 7^3 + 7^iをM回計算する

方針：
x = N^iをM回加算し総和を求める
x ≦ 10**9の場合は x, x > 10**9より大きい場合は'inf'を出力
1e9 = 10**9
'''

n, m = map(int, input().split()); x = 1
for i in range(1, m + 1):
  x += n**i
print(x if x <= 1e9 else 'inf')

'''
問題B_399：Ranking with Ties・・・解けない
時間計算量：O(N^2)

N 人それぞれの順位を出力
問題の言い換えができないか
＝もっと楽に実装できないか検討

'''

# 別解
n = int(input())
p = list(map(int, input().split()))
for i in range(n):
  rank = 1
  for j in range(n):
    if p[j] > p[i]:
      rank += 1
  print(rank)

'''
問題B_398：Full House 3・・・解けた
ロジックの整理をする
時間計算量：O(N)

与えられた整数がフルハウスになるかを求める
フルハウスの定義
7枚のカードの内3枚以上同じペアが1組、2枚以上の同じペアが1組いること

方針：
数列の中の重複数をカウントし
大きい順に並べ替えて、
0番目が3以上かつ1番目が2以上であれば、フルハウスと言える

セットを使い重複を除き数列の要素の件数分ループ
取り出された要素のカウント件数を数列リストから順に検索しリストへ追加
ループ完了後降順に並べかえる


・リストの0番目が3以上かつ1番目が2以上の場合：Yes
・リストが1件の場合：No
・それ以外：No

'''

c = list(map(int, input().split()))
c_set = set(c); f_h = []; s = 'No'
for a_i in c_set:
  x = c.count(a_i)
  f_h.append(x)
f_h.sort(reverse=True)

if len(f_h) == 1:
  print(s)
  exit()
elif f_h[0] >= 3 and f_h[1] >= 2:
  s = 'Yes'
print(s)

'''
問題B_397：Ticket Gate Log


S は i, o からなる長さ 1 以上100 以下の文字列
文字列の長さが偶数であり、奇数文字目が i で偶数文字目が o になる
i, oの挿入回数を求める

奇数文字目がiで偶数文字目がoである偶数の文字列であること

方針：
先頭から順にi, i+1番目の文字列を取得し、
取得した文字列が'io'ではない場合に挿入件数をカウントする

'io'ではない場合：
3パターン
  2文字同じ：'ii', 'oo'
  2文字で逆：'oi'
  または1文字：o, i

偶数：[0, 2, 4, 6, 8, 10] 2n
奇数：[1, 3, 5, 7, 9, 11] 2n+1


偶数でoの場合
奇数でiの場合
'''
s = list(input()); cnt,k = 0, 0
target = 'i'; ans = 0
for c in s:
  if target == c:
    target = 'o' if target == 'i' else 'i'
  else:
    ans += 1
if target == 'o':
  ans += 1
print(ans)

'''
問題B_396：Card Pile・・・解けた

方針：
クエリーのタイプ1,2に分けて処理を行う
1の場合：整数xをリストの先頭に追加する
2の場合：リストの先頭から整数を取り出し、出力し取り出した要素を削除
タイプが2の時のみ出力される
'''

q = int(input())
card = [0] * 100
for i in range(q):
  type = list(map(int, input().split()))
  if type == 1:
    card.insert(0, type[1])
  elif type == 2:
    x = card[0]
    card.pop(0)
    print(x)

'''
問題B_395：Make Target・・・解けない（グリッドが苦手）

方針：
まずは問題文の通りに実装する

グリッド N*Nを出力する問題
i=行 j=列

# = 黒
. = 白

j=N+1-i
i ≦ j でかつiが奇数ならば黒, 偶数なら白
※この時すでに塗られている色は上書きする
  マス(i, i)=左上
  マス(j, j)=右下

i > j
 何もしない

 
最初にN**2の黒マスのグリッド（二次元リスト）を作成
'''

# 結局このプログラムはにをしたい？
# 外から順に周りが内へ黒白黒白と交互に塗られたN*Nのグリッド模様を出力したい
# N * N = N^2（行 * 列）＝2重ループ
# (i, j) = 行、列＝２次元配列
# 最初は白マスの二次元配列のみある
# 操作 i = 1-Nの順に操作を行う
# 条件
# j = N + 1 - i？？？
# i ≦ jの場合（行列の繰り返し処理に入る）
# iが奇数ならば黒（#）偶数ならば白（.）マスを塗りつぶす
# （i > j場合は何もしない）
# N*(N^2) = 5回 * 25回 = 125回
# 

# 自分なりの咀嚼
# i <= j 行が列を超えていない

n = int(input())
grid = [['.'] * n for _ in range(n)]

for i in range(n):
  j = n - i - 1 # 列の範囲 -1することで外から内へ行列の範囲を狭めている
  if i <= j: # 行が列を越えるまで
    for x in range(i, j + 1): # 0から5 5回
      for y in range(i, j + 1): # 0から5 5回
        if i % 2 == 0:
          grid[x][y] = '#'
        else:
          grid[x][y] = '.'

for row in grid:
  print(''.join(row))

'''
問題B_394：cat・・・解けた

時間計算量：O(N)O(N)
文字列を昇順に並べかえる

方針：
文字列の長さをindexとしてans配列へ要素を保存する
文字列の長さが小さい方が順になるためソートも不要


昇順：
配列[文字列の長さ] = 要素で保存

降順：
配列[文字列の長さ] = 要素で保存
配列[::-1]逆順または配列[文字列の長さ - 1]でOK

'''
n = int(input())
ans = [''] * n*50
s = [input() for _ in range(n)]
for i in range(n):
  ans[len(s[i])] = s[i]
print(*ans, sep='')

'''
問題B_394：A..B..C・・・解けない
時間計算量：O(N^3)
等間隔判定

3重ループで回す→発想は間違っていない
1 ≦ i < j < k ≦ |S|

方針：
以下の条件を満たすときのカウントする
・j - i = k - j
・i番目が'A'かつj番目が'B'かつk番目が'C'

以下の場合は処理をスキップしカウントしない
i >= j or j >= k
i != 'A' or j != 'B' or k != 'C'

別解
i = j - (k - j)を求める
iの回数が確定するためcj,kの2重ループで実装できる
'''

s = list(input()); cnt = 0
n = len(s)
# for i in range(n):
for j in range(n):
  for k in range(n):
    i = j - (k - j)
    if i < 0:
      continue
    # 条件が満たさない時 j - i = k - j（等間隔でない時）
    if i >= j or j >= k:
      continue
    if j - i != k - j:
      continue
    if s[i] != 'A':
      continue
    if s[j] != 'B':
      continue
    if s[k] != 'C':
      continue
    cnt += 1
print(cnt)

'''
問題B_392：Who is Missing?・・・解けた

時間計算量：O(N)
N=10
(N+1)/2 → (10 + 1) / 2 = 5.5（約5-6回） 
方針：
Xの整数を含まない1以上N以下の整数をリストへ追加し出力
'''
n,m = map(int, input().split())
a = list(map(int, input().split()))
ans = [num for num in range(1, n+1) if num not in a]
print(len(ans))
print(*ans)

'''
問題B_392：Seek Grid・・・解けない
時間計算量：N(N^4)

全探索によってM*Mマスと一致する(a, b)の座標を求める

＊Sの中にTのマスが含まれるかを探す
1 ≦ a, b ≦ N - M + 1を満たす座標を見つける→1箇所だけ現れるa, bの座標
全探索の範囲をN-M+1にすることでa, bの座標の位置を固定している？

S=N*N,T=M*M 2つのグリッド
(i, j) = 行、列

Sij 1 ≦i,j ≦ N
Tij 1 ≦i,j ≦ M

'.'・・・白
'#'・・・黒



条件
すべてのi, j(1 ≦ i,j ≦ M)
Sa+i - 1, b+j-1 = Tij

条件を満たすa,b(座標は)ちょうど1組存在する


'''

n,m = map(int, input().split())
s = [input() for _ in range(n)]
t = [input() for _ in range(m)]

for a in range(n- m + 1):
  for b in range(n - m + 1):
    ok = True
    for i in range(m):
      for j in range(m):
        if s[a + i] [b + j] != t[i][j]:
          ok = False
    if ok:
      print(a + 1, b + 1)

'''
問題B_390：Geometric Sequence・・・等比数列（解けない）

等比数列とは
隣り合う項の比が一定
つまり、
＊後ろの数 / 前の数 = 比
↓
Ai, Ai+1, Ai+2, Ai+3, Ai+4

＊Ai+1/Ai と言い換えられる

＊Ai+1/Ai = Ai+2/Ai+1と等しい
分数だと計算に誤差が出るため値を移行して

これが等比数列の計算式：
＊A^2i+1 = AiAi+2
(1 ≦ i ≦ N ≦ N-1) = 1 ≦ i ≦ N ≦ N-2

'''

n = int(input())
a = list(map(int, input().split()))
ok = True
for i in range(n - 2):
  if a[i]*a[i+2] != a[i+1]**2:
    ok = False
    break
print('Yes' if ok else 'No')


'''
問題B_389：tcaF・・・階乗の計算・・・解けた


時間計算量：O(N)
方針
i*i+1*i+2..==xと一致するまで繰り返す
n! = n * (n-1)*(n-2)*(n-3)...3 * 2 * 1

N個のものを並べるときの考え方
n * (n - 1)がxと一致するかを判定する
'''

x = int(input()); ans = 1
for i in range(2, 21):
  ans *= i
  if ans == x:
    print(i)
    break
  elif ans > x:
    break

'''
問題B_388：Heavy Snake・・・解けた

シンプルに考える

0,1番目は固定
データ構造を使うときは本当に必要かを考える
無駄なコードを書かない
'''
n, d = map(int, input().split())
data = [list(map(int, input().split())) for _ in range(n)]

for k in range(d):
  c = k + 1
  max_value = float('-inf') 
  for i in range(n):
      value = data[i][0] * (data[i][1] + c) 
      max_value = max(max_value, value)
  print(max_value)

