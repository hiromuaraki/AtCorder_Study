from math import ceil
from random import randint
'''
二分探索法（バイナリサーチ）
時間計算量：O(log n)
期待値：検索値のindex番号
性質：半分に絞る

＊真ん中を基準として探索範囲を前後に絞る方法

指数関数的に仕事量が下がっていく
前；left = 0 後ろ：right = 配列の長さ - 1
中央：mid
中央値＝(left + right) // 2

イメージ：
leftは右に、rightは左に都度移動していく
繰り返し条件：
前が後ろとすれ違わない間繰り返す
※先頭が後ろを追い越した場合は要素が存在しない時
'''

# ハッシュ関数でハッシュ値を計算
def calc_hash(data: int, length: int) -> int:
  return data % length

# 【テスト用】1-Nまでのランダムな整数な整数を生成する
def n_random(n: int) -> list:
  return [randint(1, n) for _ in range(n)]

# 中央値を計算して探索範囲を前後の半分に絞り
# 対象データの添え字を特定する
array_a = [11, 13, 17, 19, 23, 29, 31] # 期待値：5
n = 17
def binary_search(array: list, x: int) -> int:
  left, right = 0, len(array) - 1
  # 先頭が後ろとすれ違うまで繰り返す
  count = 1
  while left <= right:
    print(count)
    count += 1
    mid = (left + right) // 2
    # 要素が一致した場合はその時点でindexを返し終了
    if array[mid] == x:
      return mid
    # 真ん中の要素の方が小さい場合は 後ろ半分に探索範囲を絞る
    if array[mid] < x:
      left = mid + 1
    # 真ん中の要素の方が大きい場合は 前半分に探索範囲を絞る
    if array[mid] > x:
      right = mid - 1
  return None

print(f'index番号＝{binary_search(array_a, n)}')

'''
ハッシュ探索法
時間計算量：O(1)
期待値：[[(値1, 値2, 値3)], [(値1, 値2, 値3)], [()], [()]]
期待値：[N1]または[N1, N2, N3]または[]
性質：ハッシュ値

＊あらかじめハッシュ化した上で対象データをハッシュ値で求めることで
 確実にデータを見つける方法

データ数に関係なく仕事量は常に一定
ハッシュ値で対象データを一発で特定する

＊ハッシュ値＝データ値 % 配列の長さ * 1.5(または2)
データ数 / データサイズ = 0.64以下になるのが慣習（ハッシュテーブル検索時に影響がない）
例）9 / 14 = 0.64のためOK

ハッシュテーブルを作成
要素、indexをペアに格納し重複したデータを同じハッシュスロットに格納していく
ハッシュ化された配列からハッシュ値で対象データを特定する
'''
array_d = [12 ,25, 36, 25, 20, 30, 8, 25, 42]
x = 25
# ハッシュ化した配列を作成しハッシュ値ごとに要素を格納する 
def hash_store(array: list) -> list:
  k = ceil(len(array) * 1.5) # ハッシュ枠の計算（小数点切り上げ）
  hash_table = [[] for _ in range(k)] # ハッシュテーブルの作成 [[], [], [], []]
  # (data, index)のペアをハッシュ値ごとにハッシュテーブルへ格納していく
  for i, data in enumerate(array):
    hash_table[data % k].append((data, i))
  return hash_table

# ハッシュ値より対象データのindex番号を特定する
# 対象データ[12 ,25, 36, 25, 20, 30, 8, 25, 42]のindex番号
def hash_search(hash_table: list, x: int) -> list:
  k = x % len(hash_table) # ハッシュ値の計算
  # ハッシュ値よりxのindex番号を特定する
  return [ index for value, index in hash_table[k] if value == x]

hash_table = hash_store(array_d)
print(hash_table)
print(hash_search(hash_table, x))

'''
単純選択法（選択ソート）
時間計算量：O(n^2)
計算回数：N(N - 1) / 2
期待値：[10, 11 ,12, 13, 14]または[14, 13, 12, 11, 10]
性質：常に先頭のデータと最小値（または最大値）が入れ替わる

＊最小値または最大値を探し出し先頭と入れ替える処理を
 リストの末尾まで繰り返す方法

先頭＝index_min 最小値の添え字を管理
毎週先頭が変わる
比較＝右へ移動していく k

変数iで入れ替えた回数を管理 i = 1
変数kですべての要素を比較していく k + i
'''

array_b = [12, 13, 11, 14, 10]

# 最小値を基準とし先頭と入れ替え昇順に並び替える
def select_sort(array: list) -> list:
  i = 0
  # すべての要素を入れ替えるまで繰り返す
  while i < len(array) -1: # 4回
    index_min = i # 暫定の最小値
    k = i + 1 # 比較する要素
    while k < len(array): # 4回
      # 最小値が要素より大きい場合のみ先頭と入れ替える
      if array[index_min] > array[k]:
        w = array[index_min]
        array[index_min] = array[k]
        array[k] = w
      k += 1 # 右へ移動する
    i += 1 # 入れ替え回数を記録する
  return array

print(select_sort(array_b))

'''
単純交換法（バブルソート）
時間計算量：O(n^2)
計算回数：N(N - 1) / 2
例）N＝11
11 * 10 / 2 = 110 / 2 = 55回

期待値：[10, 11 ,12, 13, 14]または[14, 13, 12, 11, 10]
性質：隣り同士の大小の比較, 末尾から左へ移動していく

＊隣り同士の大小を比較し逆順の場合に入れ替える処理を繰り返す方法

i＝配列の長さ - 1
末尾＝i 手前＝i - 1
'''

array_c = [10, 9, 8, 7, 1, 2, 4, 6, 0, 3, 5]

# 隣り同士の大小を比較し逆順の場合昇順に並べ替える
def bubble_sort(array: list) -> list:
  k = 0
  # すべて並べ替えるまで繰り返す
  while k < len(array):
    i = len(array) - 1
    while i > k:
      # 隣り同士の大小を比較し要素が逆順なら入れ替える
      if array[i- 1] > array[i]:
        w = array[i - 1]
        array[i - 1] = array[i]
        array[i] = w
      i -= 1 # 比較要素を移動する
    k += 1 # 入れ替え回数を記録する
  return array

print(bubble_sort(array_c))

'''
単純挿入法（挿入ソート）
時間計算量：O(n^2)
期待：[5, 3 ,4, 1, 2]または[1, 2, 3, 4, 5]
性質：整列/未整列のデータに分ける
（空き要素枠へデータを挿入する）

＊既に整列済みのデータ群に対して適切な位置にデータを挿入していくソート方法

先頭の要素はすでにソート済みとする
離れた位置同士の要素を比較していく
'''

array_e = [5, 3, 4, 1, 2]

'''
[3, 5, 4, 1, 2] k = 1 i = 0 x = 3
k = 0
[3, 5, 4, 1, 2] k = 2 i = 2 x = 4
'''

# 整列済みと未整列データの大小を比較しデータを正しい位置に挿入する
def insert_sort(array: list) -> list:
  i = 1
  # すべての要素が整列するまで繰り返す
  while i < len(array):
    k = i
    x = array[i] # 挿入するデータを準備
    # すべての要素を比較するまで繰り返す
    while k > 0 and array[k - 1] > x:
      array[k] = array[k - 1]
      k -= 1 # k番目を空き要素にする
    array[k] = x # データを挿入する
    i += 1 # 整列した回数を記録する
  return array

print(insert_sort(array_e))

'''
クイックソート
時間計算量：O(n log n)
期待値：昇順（または降順）に並べ替えたデータ
性質：基準値を決めて大小にデータを分割する

基準値＝配列の長さ / 2
基準値より小さい場合は並び替える範囲を大グループに分割
基準値より大きい場合は並び替える範囲を小グループに分割

先頭＝left 後ろ＝right 基準値＝(left + right) / 2
変数 i ・・・右へ移動していく（i + 1）
変数 k ・・・左へ移動していく（k - 1)

i < k・・・先頭が後ろとすれ違うまで繰り返す（外側）

条件：
先頭が後ろとすれ違っていなければ基準値よりも前に要素を移動していく
基準値の方が大きければ真ん中と入れ替える

再帰処理：
先頭の添え字がまだ真ん中の添え字 - 1未満であれば
後ろの添え字がk + 1未満であれば
'''
array_f = [5, 4, 7, 6, 8, 3, 1, 2, 9]

'''
[1, 2, 3, 6, 8, 4, 7, 5, 9] i = 1 k = 1 left = 0 right = 1

'''

# 基準値を決めて大小のグループに分割し昇順に並び替える
def quick_sort(array: list, left: int, right: int) -> list:
  i, k = left + 1, right
  # 先頭が後ろとすれ違う（真ん中に来る）まで繰り返す
  while i < k:
    # リストの末尾まで大きい要素を探し出す
    while array[i] < array[left] and i < right:
      i += 1
    # リストの先頭まで小さい要素を探し出す
    while array[k] >= array[left] and k > left:
      k -= 1
    # 先頭と後ろ中央まで来ていなければ基準値を入れ替える
    if i < k:
      # 小さいデータと大きいデータの位置を入れ替える
      w = array[i]
      array[i] = array[k]
      array[k] = w

  # 真ん中と基準値を入れ替える
  if array[left] > array[k]:
    w = array[left]
    array[left] = array[k]
    array[k] = w

  # 再帰処理
  if left < k - 1:
    quick_sort(array, left, k - 1) # 真ん中の確定したデータを除いた範囲
  if k + 1 < right:
    quick_sort(array, k + 1, right)
  
  return array

print(quick_sort(array_f, 0, len(array_f) - 1))

array_g = [5, 4, 7, 6, 8, 3, 1, 2, 9, 2, 10]

'''
クイックソート
基準値を決めて大小のデータに分ける処理を繰り返し昇順に並べ替える

i・・・基準値よりも大きい要素探し出す（右へ移動）
k・・・基準値よりも小さい要素を探し出す（左へ移動）
left・・・先頭
right・・・末尾
'''

def quick_sort2(array: list, left: int, right) -> list:
  i, k = left + 1, right
  # 先頭と末尾がすれ違うまで繰り返す
  while i < k:
    # 基準値よりも大きいデータを末尾まで探す
    while array[i] < array[left] and i < right:
      i += 1 # 右へ移動する
    # 基準値よりも小さいデータを先頭まで探す
    while array[k] >= array[left] and k > left:
      k -= 1 # 左へ移動する

    # 先頭と末尾がすれ違っていない場合のみ大小を入れ替える
    if i < k:
      w = array[i]
      array[i] = array[k]
      array[k] = w
  
  # 基準値が変数kよりも大きい場合のみ入れ替える
  if array[left] > array[k]:
    w = array[left]
    array[left] = array[k]
    array[k] = w
  # 再帰処理
  # 先頭からk - 1した範囲を渡しクイックソートを繰り返す
  if left < k - 1:
    quick_sort2(array, left, k - 1)
  # 末尾からk + 1の確定した範囲を渡しクイックソートを繰り返す
  if k + 1 < right:
    quick_sort2(array, k + 1, right)
  return array

print(quick_sort2(array_g, 0, len(array_g) -1))

'''
クイックソート（Python Ver）
再帰処理が理解できないと使ってはダメ

初回：
left：[1, 2, 2]
middle：[3]
right：[5, 4, 7, 6, 8, 9, 10]

再帰処理：
1回目
left：[1]
middle：[2, 2]
right：[]

2回目：
left：[1]
middle：[2]
right：[]

＊return [1] + [3] + [2, 2] + [] = [1, 2, 2, 3]

3回目：
left：[5, 4]
middle：[6]
right：[7, 8, 9, 10]

4回目：
left：[]
middle：[4]
right：[5]

＊return [] + [4] + [5] + [6] [4, 5, 6] 

5回目:
left：[7, 8]
middle：[9]
right：[10]

6回目：
left：[7]
middle：[8]
right：[]

＊return [7] + [8] + [] + [9] + [10] + [7, 8, 9, 10]
[1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10]
'''

array_h = [5, 4, 7, 6, 8, 3, 1, 2, 9, 2, 10]

def quick_sort_sub(array: list) -> list:
  # 再帰処理修了条件（必須！！）
  if len(array) <= 1:
    return array

  pivot = array[len(array) // 2]
  left   = [x for x in array if x < pivot] # [1, 2, 2]
  middle = [x for x in array if x == pivot] # [3]
  right  = [x for x in array if x > pivot] # [5, 4, 7, 6, 8, 9, 10]
  return quick_sort_sub(left) + middle + quick_sort_sub(right)

print(quick_sort_sub(array_h))

'''
単純挿入法（挿入ソート）

未整列データを正しい位置に挿入していく処理を繰り返し
並べ替えをする方法

整列済み/未整列のデータの大小を比較する
整列済み＝i
未整列＝k
挿入データ＝x

'''

array_i = [5, 2, 3, 1, 4]

'''
<机上デバッグ>
1回目：
x = 2
[5, 5, 3, 1, 4] k > 0 かつ 5 > 2 T k = 0
[2, 5, 3, 1, 4] i = 1 i < 配列の長さ T

２回目:
x = 3 k > 0 かつ 5 > 3 T k = 1
[2, 5, 5, 1, 4] 2 > 3 F i < k
[2, 3, 5, 1, 4] i = 2 i < 配列の長さ T

3回目：
x = 1
[2, 3, 5, 5, 4] k > 0 かつ 5 > 1 T k = 2
[2, 3, 3, 5, 4] k > 0 かつ 3 > 1 T k = 1
[2, 2, 3, 5, 4] k > 0 かつ 2 > 1 T k = 0
[1, 2, 3, 5, 4] i = 3 i < 配列の長さ T

4回目：
x = 4
[1, 2, 3, 5, 5] k > 0 かつ 5 > 4 T k = 3
[1, 2, 3, 5, 5] k > 0 かつ 3 > 4 F k = 3（操作なし）
[1, 2, 3, 4, 5] i = 4 i < 配列の長さ F

[1, 2, 3, 4, 5]
'''
def insert_sort_sub(array: list) -> list:
  i = 1 # 先頭データは整列済み
  # 未整列のデータを並べ替えるまで繰り返す
  while i < len(array):
    k = i # 未整列データを参照する変数
    x = array[i] # 挿入データ
    # 未整列データが0件より多いかつ整列済みデータの方が挿入データ（手前）よりも大きい場合のみ
    while k > 0 and array[k - 1] > x:
      array[k] = array[k - 1] # 整列済みデータ→未整列データ
      k -= 1 # k番目を空き要素にする
    array[k] = x # 挿入データ→k
    i += 1 # 次の挿入データの参照先へ変更する
  return array

print(insert_sort_sub(array_i))


'''
単純交換法（バブルソート）

隣り度同士データの大小を比較し逆順の場合、
入れ替えることを繰り返すソート方法

k＝入れ替え回数の管理
i＝後ろの要素
i - 1＝手前の要素

'''
array_j = [5, 3, 4, 1, 2, 10, 11, 14, 17, 20]

def bubble_sort_sub(array: list) -> list:
  k, length = 0, len(array) - 1
  # すべての要素が並べ替え終わるまで繰り返す
  while k < length - 1:
    i = length
    # すべての要素を比較するまで繰り返す
    while i > k:
    # 隣り同士の要素の大小を比較し手前が大きい時のみ要素を入れ替える
      if array[i - 1] > array[i]:
        w = array[i - 1]
        array[i - 1] = array[i]
        array[i] = w
      i -= 1 # 左へ移動する
    
    k += 1 # 入れ替え回数を+1加算する
  return array

print(bubble_sort_sub(array_j))

'''
単純選択法（選択ソート）
最小値（または最大値）を探し出し常に先頭の要素と入れ替える処理を繰り返し並べ替える
ソート方法

index_min＝最小値を管理
i＝先頭のデータ、入れ替えが完了した際に要素を確定させる
k＝比較していく要素

最小値（手前）の方が要素よりも大きければindex_minを更新

先頭のデータと最小値を入れ替える
'''
array_k = [12, 13, 11, 12, 14, 13, 10]

'''
<机上デバッグ>
1回目：
i = 0
k = i + 1
index_min = i・・・暫定最小値を設定
[12, 13, 11, 12, 14, 13, 10] 
[10, 13, 11, 12, 14, 13, 12] 0番目の最小値が確定
[10, 11, 13, 12, 14, 13, 12] 1番目の最小値が確定
[10, 11, 12, 13, 14, 13, 12] 2番目の最小値が確定
[10, 11, 12, 12, 14, 13, 13] 3番目の最小値が確定
[10, 11, 12, 12, 13, 14, 13] 4番目の最小値が確定
[10, 11, 12, 12, 13, 13, 14] 5番目の最小値が確定 6番目は並べ替え不要
'''


array_k = [12, 13, 11, 12, 14, 13, 10]

def select_sort_sub(array: list) -> list:
  i = 0
  # 先頭を除いた要素の件数分繰り返す
  while i < len(array) - 1:
    index_min = i # 最小値の設定
    k = i + 1 # 比較要素のインデックスを設定
    # すべての要素を比較するまで繰り返す
    while k < len(array):
    # 要素の方が最小値より小さい場合のみ最小値を更新する
      if array[k] < array[index_min]:
        index_min = k
      k += 1 # 要素を次へ進める
    # 先頭のデータと最小値を入れ替える
    w = array[i]
    array[i] = array[index_min]
    array[index_min] = w
    i += 1
  return array

print(select_sort_sub(array_k))


array_l = [12, 13, 11, 10, 12, 14, 13, 10]
# 選択ソート（Python版）
# 最小値を検索し格納した結果を返す
# 時間計算量：O(N^2) 計算回数：N(N - 1) /2 = 7 * 6 / 2 = 21回
# シンプルに見えるが既存のアルゴリズムのコードの方が適切
def select_sort_sub_sub(array: list) -> list:
  n, result = len(array), []
  array_c = array.copy() # リストをコピーするためメモリ消費が大きい
  for _ in range(n):
    min_val = min(array_c) # 最小値を検索
    result.append(min_val) # 最小値をリストに追加
    array_c.remove(min_val) # 最小値を削除（重複した要素が複数ある場合の担保が難しい、処理が遅い）
  return result

print(select_sort_sub_sub(array_l))
print(array_l)

# テスト用　1-N個のランダムな整数リストを作る
# test_data = n_random(10000)

'''
ハッシュ探索
時間計算量：O(1)
計算回数：1

あらかじめハッシュ化されたデータ群の中からハッシュ関数で計算した
ハッシュ値より対象データを見つけ出す方法

ハッシュ関数＝値 % 配列の長さ

1.ハッシュテーブルを作成しハッシュ値ごとに要素を格納する
2.ハッシュ関数よりハッシュテーブルより対象データを見つけ出す

重複データは同じスロットに格納する＝チェイニング手法 チェーンのように横に繋がるイメージ
※チェイニング手法ではない場合はハッシュスロット（配列の長さ * 1.5 or 2）の空きを用意する必要あり
'''
array_m = [12 ,25, 36, 25, 20, 30, 8, 25, 42]
x = 25
# ハッシュ関数よりハッシュテーブルを作成しハッシュ値ごとに要素を格納
def hash_store_sub(array: list) -> list:
  length = len(array)
  if length <= 0:
    return
  hash_table = [[] for _ in range(length)] # ハッシュテーブルを作成 [[], [], []]
  # ハッシュ関数よりハッシュ値を計算
  for index, data in enumerate(array):
    hash_table[calc_hash(data, length)].append((data, index)) # キー：値/値：index ペアで格納 [[(), (), ()], [()], [()]]
  return hash_table

# ハッシュテーブルより対象データを見つけ出す
def hash_search_sub(hash_table: list, x: int) -> int:
  k = calc_hash(x, len(hash_table))
  return [index for value, index in hash_table[k] if value == x]

hash_table_2 = hash_store_sub(array_m)
print(hash_table_2)
print(hash_search_sub(hash_table_2, x))

random_data = n_random(10000)
print(hash_store_sub(random_data))

'''
ハッシュスロット * 1.5個分用意したver
ハッシュ値がバッティングした場合は空き要素へ移動する
'''
array_n = [12, 25, 36, 20, 30, 8, 42]

def hash_store_h(array: list) -> list:
  length = ceil(len(array) * 1.5)
  hash_table = [0 for _ in range(length)] # ハッシュスロットを0で初期化
  for data in array:
    k = calc_hash(data, length) # ハッシュ値を計算
    # 空き要素が見つかるまで繰り返す
    while hash_table[k] != 0:
      k = (k + 1) % length # *kが配列の長さを超えても余りが0となり先頭に戻る
    hash_table[k] = data # ハッシュテーブルへ要素を格納
  return hash_table

def hash_search_h(hash_table: list, x: int) -> int:
  length = len(hash_table)
  k = calc_hash(x, length)
  # データが格納されていない要素が見つかったら繰り返しを終了
  while hash_table[k] != 0:
    if hash_table[k] == x:
      return k
    else:
      k = (k + 1) % length
  return k
    

array_D = hash_store_h(array_n)
print(array_D)
print(f'格納場所は{hash_search_h(array_D, 36)}番目の要素です')

'''
二分探索法（バイナリサーチ）
時間計算量：O(log N)

中央値を計算しデータを大小に分け探索範囲を絞り込む方法

中央値＝（先頭 + 末尾）// 2
left = 先頭 right = 末尾
'''
array_o = [11, 13, 17, 19, 23, 29, 31]

def binary_search_sub(array: list, x) -> int:
  left, right = 0, len(array) - 1
  # 先頭と末尾がすれ違うまで繰り返す
  while left <= right:
    mid = (left + right) // 2 # 中央値を計算
    # 検索データが一致したらindexを返す
    if array[mid] == x:
      return mid
    # 先頭に探索範囲を絞る
    elif array[mid] > x:
      right =  mid - 1
    # 末尾に探索範囲を絞る
    else:
      left = mid + 1
  return None

print(binary_search_sub(array_o, 23))