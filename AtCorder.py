# AtCorder演習問題集

'''
問題A：正方形の面積
'''

def input_split() -> list:
  return input().split()

def square(x: int) -> int:
  return x**2

print(square(int(input())))


'''
問題A：Linear Search
xがnに含まれているか否か
含まれている：Yes
含まれていない：No
'''

def contain(x: list, n: list) -> str:
  return 'Yes' if x[1] in n else 'No'

x = input()
n = input()
print(contain(x.split(), n.split()))


'''
問題A：２枚のカード
赤いカード1枚＋蒼カード1枚選ぶ
2枚のカードに書かれた数の合計がk枚になるか

p[i] + q[i] = k
q = k - p[i]
'''

def two_cards(k: list, p: list, q: list) -> str:
  for i in range(len(p)):
    diff = int(k[1]) - p[i]
    if diff in q:
      return 'Yes'
  return 'No'

'''
文字列リスト→整数リストへ変換
'''
def str_int(l: list) -> list:
  return [int(n) for n in l]


k, p, q = input('数字を２桁>>'), input('数字を５桁>>'), input('数字を５桁>>')
red_cards = str_int(p.split())
blue_cards = str_int(q.split())

print(two_cards(k.split(), p.split(), q.split()))

'''
問題A：Binary Representation 1

漸化式（芋づる式）
例）
13 ÷ 2 = 6 余り 1
6  ÷ 2 = 3 余り 0
3  ÷ 2 = 1 余り 1
1  ÷ 2 = 0 余り 0

以降0桁
0  ÷ 2 = 0 余り 0を繰り返す

10進法を2進法に変換(0,1の2種類の数字だけで表す)

1. 10進数を2で割る
2. 余りを記録する
3. 商が0になるまで以上を繰り返す
4. 逆順にしてバイナリ文字列を返す

整数は最大10家まで（0埋め）
'''
def binary_representation(n: int, binary='') -> str:
  if n // 2 == 0 and len(binary) > 9:
    return binary[::-1]
  binary += str(n % 2) # 余りを記録
  return binary_representation(n // 2, binary)

print(binary_representation(int(input())))

'''
問題A：来場者数（WA）
アルゴリズム見直す必要あり
'''

def how_many_guests(info: list, guests: list) -> None:
  if int(info[1]) != len(guests):
    return
  total = ''
  for _ in range(int(info[1])):
    slice = input().split()
    g_l = map(int, guests[int(slice[0]) -1 : int(slice[1])])
    total += str(sum(g_l)) + '\n'
  print(total)

info = input().split()
guests = input().split()
how_many_guests(info, guests)

'''
問題A：BinarySearch（二分探索法）

配列全体を探索範囲とする
探索範囲の中央とxを比較する。この結果に応じて次の操作を行う

・同じである場合、この時点でYesが決まる →検索要素と一致するか
・中央の方が小さい場合：探索範囲を後半部分に絞る → 大小関係の比較
・中央の方が大きい場合：探索範囲を前半部分に絞る → 大小関係の比較
探索範囲がなくなってもYesであることが決まらない場合、答えはNo

配列の順番を昇順にする
検索範囲を中央で分割するため、head, tail, centerを用意
(head + tail) // 2 = center(中央)平均値を求める

head >= tailより大きくなることはあり得ない
→検索要素が存在しない
'''

def binarySearch(x: list, array: list) -> int:
  head, tail, center = 0, int(x[0]), 0
  while head <= tail: # 目的のデータが見つかるまで探索する
    center = (head + tail) // 2 # 中央に分割する
    # 同じである場合この時点で探索を終了する
    if array[center] == int(x[1]):
      return center + 1

    # 中央の方が小さい場合：探索範囲を後半部分に絞る
    if array[center] < int(x[1]):
        head = center + 1
    # 中央の方が大きい場合：探索範囲を前半部分に絞る
    else:
        tail = center - 1
  return 0

search = input().split()
array = map(int, input().split())

print(binarySearch(search, list(array)))




'''
問題A：Printer
N = プリンターの台数
K = チラシの印刷枚数
Ai = 秒数（時間）各プリンターでごとで異なる

Ai秒ごとにチラシが1枚印刷される
二分探索法を使う

時刻（t）を時刻を求める
条件（枚数 ≥ K）を満たす最小のtを二分探索で探す。

'''
def printer(n_k: list, a: list) -> int:
  right, left, total = 0, int(n_k[1]) - 1, 0
  while left <= right:
    mid = (left + right) // 2
    # このロジックを確認する
    total = sum(mid // ai for ai in a)  # 時刻midまでに印刷されるチラシの合計
    if total >= int(n_k[1]):
      right = mid - 1
    else:
      left = mid + 1
  return left

n_k = input().split()
a = map(int, input().split())
print(int(printer(n_k, list(a))))











