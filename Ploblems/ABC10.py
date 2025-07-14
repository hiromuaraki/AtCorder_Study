'''
AtCorder ABC問題10問
'''
# print(placing_marbles('101'))
# print(placing_marbles('100'))
# print(placing_marbles('001'))
# print(placing_marbles('010'))
# print(placing_marbles('111'))


'''
問題C_042：こだわり者いろはちゃん・・・勉強？？（分からない）
'''
def iroha_c(N, K, D) -> int:
  # N から順にチェック
  i = N
  while True:
    # 数 i の各桁をチェック
    num = i
    valid = True
    while num > 0:
      digit = num % 10  # 最下位の桁を取得
      if digit in D:  # 嫌いな桁が含まれていたら無効
          valid = False
          break
      num //= 10  # 次の桁へ
    if valid:
      print(i)  # 条件を満たす最小の数を出力
      break
    i += 1

N, K = map(int, input().split())
D = set(map(int, input().split()))  # 嫌いな桁をセットで管理
iroha_c(N, K, D)

'''
8. 問題C_085：:Otoshidama（300点）・・・解説見る
時間計算量：O(N^2)

データ構造：タプル
A + B + C = N
C = N - A - B
'''

def otoshidama(n: int, y: int) -> None:
  for a in range(n + 1): # 0-8（9回）
    for b in range(n - a + 1): # 0-8（9回）
      c = n - a - b # a, bの枚数が決まればcの枚数が分かる
      total = 10000*a + 5000*b + 1000*c
      if total == y:
        print(a, b, c)
        exit()
  print(-1, -1, -1)

# otoshidama(9, 45000)
# otoshidama(20, 196000)
# otoshidama(1000, 1234000)
# otoshidama(2000, 20000000)


'''
問題C_049：白昼夢
S＝T
「文字列を、先頭から4つの単語で切り出して、全部使い切れるか？」

erase|dream
文字列を分割する（範囲指定し文字列をスライスで取得する）
文字列の長さ分繰り返し空文字列のTに一致した文字列を連結させる
'''

S = input()
i = 0
words = ["dream", "dreamer", "erase", "eraser"]
# 文字列の最後まで一致するか確認する
while i < len(S):
  matched = False
  for word in words:
    if i + len(word) <= len(S) and S[i:i+len(word)] == word:
        i += len(word)
        matched = True
        break
  if not matched:
      print("NO")
      exit()
print("YES" if i == len(S) else "NO")

'''
問題C_407：Security2（この問題は後で再度復習する）
時間計算量：O(N^2)

ボタンA：tの末尾に0を追加
ボタンB：tに含まれる文字列へ+1加算

入力された文字列と空文字tを一致させるさせるための
ボタンA,Bの最小回数を求める

＊右から左へ貪欲に見ていく「後ろから考える」

A：文字列の長さを1増やす
B：各文字列へ+1加算し全体的に影響する（文字列の長さは増えない）

Aの操作回数＝文字列の長さ
B*k＝Bのボタンの操作回数

＊0-9の範囲でkは循環する
ボタンA＋B*k＝答え

ボタンBの回数を毎回カウントする

S[i]番目の文字列とt[0]番目の文字列を一致させるにはk回Bボタンを押す
t[0] + k = S[i] % 10
k = (S[i] - t[0]) % 10

ボタンBが押された回数の計算
k = (S[i] - t[0]) % 10
例）
S＝2028
S[i] = 2 t[0] = 0 
k = (2 - 0) % 10
  = 2
'''

# 入力
s = input()
sum_b = 0
n = len(s)
for i in range(n-1, -1, -1):
  v = int(s[i])
  u = int(s[i+1]) if i < n-1 else 0
  b = (10 + v - u) % 10
  sum_b += b
ans = sum_b + n
print(ans)

'''
問題C_405：Sum of Product・・・勉強 解説見る
∑（シグマ）・・・数列の和＝総和を表す記号
時間計算量：O(N^2)
N(N-1) / 2

300_000 * (299_999) / 2 = 5000000....（TLE）
アルゴリズムを考える

方針：O(N)
先に全体の合計を取得し数列の回数以下の処理を繰り返す
合計-数列[i]の差を取る
解答 * 要素 + 合計との差を行い、累積和を更新していく

'''
n = int(input())
a = list(map(int, input().split()))
MOD = 1e7 # 10億7（オーバーフロー対策）
sum_a = sum(a) % MOD
ans = 0
for i in range(n):
    sum_a = (sum_a - a[i]) % MOD
    ans = (ans + a[i] * sum_a) % MOD
print(ans)

'''
問題C_402：Dislike Foods（土日やる）解説見る

N=食材の種類数, 日にち
M=料理の個数
K=料理に使われている食材の数
Bi=克服する食材の番号
'''

'''
問題C_401：K-bonacci（土日やる）解説見る
'''

'''
問題C_400：2^a b^2（土日やる）解説見る
'''