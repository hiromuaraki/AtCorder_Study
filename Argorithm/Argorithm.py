from math import ceil
from random import randint
'''
アルゴリズムとは
＊問題や課題を解決するための手順

絶対条件は２つ
・結果が得られる
・必ず終わる

大きく分けると
探索・整列・数値計算・文字列探索の４つ

＊探索
  - ＊線形探索
  - ＊二分探索
  - ＊ハッシュ探索

＊整列
  - ＊選択ソート
  - ＊バブルソート
  - ＊挿入ソート
  - ＊クイックソート
  - マージソート
  - ヒープソート
  - シェルソート

＊数値計算
  - ＊エラトステネスのふるい（素数を求める）
  - ＊ユークリッドの互除法（最大公約数を求める）
  - ガウスの消去法（連立１次方程式）
  - 台形公式（定積分の近似値を求める）
  - ダイクストラ法（グラフで最適経路を求める）
  - 二分法（方程式を解く）
  - ニュートン法（方程式を解く）

＊文字列探索
  - 力まかせの検索法（先頭から1文字ずつずらしながら探す）
  - KMP（不一致箇所に着目して探す）
  - BM（部分文字列の末尾から探す）


＊時間計算量：
プログラムがどれくらいの時間（ステップ）で終わるか


ビッグオーの記号（ランダウ）
O(N), O(1)→パソコンの仕事の量を表すマーク

O(1)・・・データがどれだけ増えても、仕事の量がずっと同じ
例）ハッシュテーブルの検索：ハッシュ値で一発で見つけられる為

O(N)・・・データが増えると、仕事の量もその分増える
データNに対して比例して仕事量も増える
例）配列を先頭から順に全て見て特定の値を探す
   リストのcountメソッド（全要素をチェックする）N個のデータでNステップ

O(log N)・・・データを半分ずつ減らす
例）二分探索法 → 探索範囲をどんどん半分に絞る

O(N^2)・・・・全部を何回もチェック
例）バブルソート N × N回

O(N long N)・・・ちょっと賢いソート
例）トランプ52枚を各グループ半分に分けた後にくっつける
クイックソートやマージソート

O(2^N)・・・・・・めっちゃ遅い
データが少し増えるだけで仕事量が倍増える
例）組合せや部分集合で出てくる

O(N!)・・・・・・・宇宙レベルで遅い
例）順列の生成
クラスの席替えを全パターン試す

<順次構造 選択構造 反復構造>

視覚的にアルゴリズムを確認するためにフローチャートを使う

企画→設計→プログラム→デバッグ→ドキュメント作成

ウォーターフォール開発だと

上流：
要件定義→基本設計(外部)→詳細設計（内部）

外部設計・・・設計の大枠、客先向け（機能設計に近いもの、フローチャート）
内部設計・・・設計の詳細、エンジニア向け（機能単位の役割など）

下流：
→実装→単体テスト（UT）→結合テスト（IT）→総合テスト→納品

単体・・・単機能のテスト
結合・・・全体の機能を流すテスト

良いプログラムとは
高速・効率・汎用的である

'''

'''
変数と配列（リスト）
int（整数）, float（小数）, str（文字列）, bool（真偽値）
整数・・・5
小数・・・5.0
文字列・・'', ""
真偽値・・・True, False（0以外はTrue）

a, b = 値1, 値2 → アンパック代入が可能

/・・・10 / 5 = 2.0（商）
//・・10 // 5 = 2（商）
%・・・10 % 5 = 0（余り）

比較演算子
a == b・・aとbは等しい
a > b・・・aはbより大きい
a < b・・・aはbより小さい（未満）
a >= b・・aはb以上（bも含む）
a <= b・・aはb以下（bも含む）

論理演算子
かつ
a and b・・・両方条件が成立する時のみTrue
(1, 1)・・・True
(0, 1)・・・False
(1, 0)・・・False
(0, 0)・・・False

または
a or b・・・両方条件が成立しない時のみFalse
(1, 0)・・・True
(0, 1)・・・True
(1, 1)・・・True
(0, 0)・・・False

否定形
等しくない、でなければ
a != b・・・aはbと等しくない
a not b・・・aはbではない

a != b・・aはb等しくない

データ構造を理解する

リスト・・・末尾から順番に格納する
[] list(Python), List<>(C#, Java)
添え字は0番目からデータを参照していく
値の追加・変更・削除が可能

ハッシュテーブル・・・キー：値のペア O(1)
辞書（Python）,HashMap(C#),Map(Java),連想配列(PHP)などが該当する

ハッシュ値で一発で検索可能
keyに配列のindex番号を設定しハッシュ値とすることも可能
valueに配列の値を設定
hash_table = {key ： value}

チェイニング管理も可能→リスト[[(), (), ()], [], [], []]の構造
重複した値も一つの集合として同じスロットに格納できる

タプル・・・



セット・・・
A = {1, 2, 3, 4}
B = {2, 3, 4, 5}

集合演算が使える
・積集合（共通部分）・・・ A & B ← (A + B) - A ∪ B = {2, 3, 4}
・和集合（＋）・・・・・・ A | B ← (A + B) - A ∩ B = {1, 2, 3, 4, 5}
・差集合（ー）・・・・・・ A - B ←  A - (B + A ∪ B = {1}
・対称差（共通部分以外）・ A ^ B ←  A ∪ B - A ∩ B =  {1 ,5}
'''

# 三角形の面積
# 底辺 × 高さ × 1/2
base, hight = 4, 6 # グローバル変数
def triangle_area(base: int, hight: int) -> int:
  return (base * hight) // 2

print(triangle_area(base, hight))

# 2つの大小を判定する
# aの方が大きい場合はa, 小さい場合はbを出力
a, b = 9, 14
def compare(a: int, b: int) -> str:
  
  if a > b:
    return str(a)
  elif a == b:
    return 'aとbは等しい'
  else:
    return str(b)

print(compare(a, b))

# 2つの変数の入れ替え
a, b, c = 11, 55, 0

c = a # aの変数をcの変数へコピー 11
a = b # aの変数をbの変数へ上書き 55
b = c # cの変数をbの変数へ上書き 11

print(a, b)

# 合計を計算するアルゴリズム
# Pythonの場合は標準関数の方が早い
array = [12, 13, 11, 14, 10]
def calcSum(l: list) -> int:
  # total = 0
  # for e in l:
  #   total += e
  # return total
  return sum(l)
  
print(calcSum(array))

# 最大値を探す
# 端から順番に2つのデータを比較していく
# O(N)
def maxSearch(array: list) ->int:
  mx = array[0] # 暫定の最大値を設定
  for i in range(len(array)):
    # 暫定の最大値が他の値より小さい時のみ中身を入れ替える
    if mx < array[i]:
      mx = array[i]
  return mx

print(f'最大値：{maxSearch(array)}')

# 線形探索法（リニアサーチ）
# 先頭から順番にひとつずつ調べて探す
# 時間計算量：O(N)
elements = [4 ,2 ,3, 5, 1]
def linear_search(array: list, keyword: int) -> list:
  return [index for index in range(len(array)) if array[index] == keyword]

# 検索したキーワードのindex番号を返す
def linear_search1(array: list, keyword: int) -> int:
  try:
    return array.index(keyword)
  except:ValueError

print(f'{linear_search1(elements, 3)}番目の要素が一致')


# ハッシュ関数でハッシュ値を計算
def calc_hash(data: int, length: int) -> int:
  return data % length

# 【テスト用】1-Nまでのランダムな整数な整数を生成する
def n_random(n: int) -> list:
  return [randint(1, n) for _ in range(n)]



'''
N!の計算（番外）
N! = n * (n-1) * (n -2)...3 * 2 * 1
例）n = 5 
5! = 5 × 4 × 3 × 2 × 1 = 120 
'''
# 階乗の計算
# 再帰処理の終了条件：0 * 1の計算は不要の為 n == 1とする
def func_x(n: int) -> int:
  return 1 if n == 1 else func_x(n - 1) * n 

'''
例）n = 5
5 + 4 + 3 + 2 + 1 = 15
'''
# nの総和
def func_sum(n: int) -> int:
  return  1 if n == 1 else func_sum(n - 1) + n

print(func_x(5))
print(func_sum(5))


'''
データの遷移


再帰処理の結果と流れ
前半はleftを引数に渡している
コールバック以降は
quick_sort_1(left)＊
[5, 4, 7, 6, 3, 1, 2]
[8]
[9]

quick_sort_2(left)＊
[5, 4, 3, 1, 2]
[6]
[7]

quick_sort_3(left)＊
[1, 2]
[3]
[5, 4]

quick_sort_4(left)＊
[1]
[2]
[]

以降は再帰処理1〜4までの結果をコールバックする

肝は middleがリストに連結されていること

quick_sort(left)
return quick_sort_4(left)
[1, 2]

quick_sort_(right)
return quick_sort_3(right)
[3, 4, 5]

quick_sort(right)
return quick_sort_2(right)
[6, 7]

quick_sort(right)
return quick_sort_1(right)
[8 ,9]

昇順の結果を返却
[1, 2] + [3, 4, 5] + [6, 7] + [8, 9]
→[1, 2, 3, 4, 5, 6, 7, 8, 9]（新しいリストを作成して返却している） 
'''

array = [11, 13, 17, 19, 23, 29, 31]
num = 17
# 二分探索法（バイナリサーチ）
# 時間計算量：O(log N)

# 中央の平均値を取り「前か・後ろか」の2つに分けて探索範囲を絞る
# ＊あらかじめデータが昇順か降順になっているデータが対象
# 中央の平均値＝３＝3回繰り返せば対象を絞れる
# leftがrightより大きくなる時は目的のデータが見つからない時
def binary_search(array: list, x: int) -> int:
  left,right = 0, len(array) - 1
  # left(前)がright(後ろ)を越える事はあり得ない
  while left <= right:
    mid = (left + right) // 2 # 中央の平均値を求める
    # 真ん中と等しかったら処理修了
    if array[mid] == x:
      return mid
    # 探す範囲を後ろに絞る
    if array[mid] < x:
      left = mid + 1
    # 探す範囲を前に絞る
    else:
      right = mid - 1
  return None

print(f'{num}は{binary_search(array, num)}番目にあります。')

# ハッシュ探索法（チェイニング法）
# 時間計算量：O(1)

# 1. ハッシュ関数でデータを格納するアルゴリズム＊
# 2. ハッシュ関数でデータを探索するアルゴリズム

# ハッシュテーブルはチェインニングで重複データを管理する

# ハッシュ値を計算する
# ハッシュ値 = データの値 % 配列の長さ
# 配列の要素数 * 1.5〜2の要素数を持つハッシュ配列を用意 0で初期化
# データの数 / データサイズ(配列の長さ * 1.5) = 0.67以下が処理に負荷がない範囲
# 添え字が被っているかチェック
# 被っていた場合：被らなくなるまでハッシュ値に1を加算し続ける
# 被っていない場合：計算済みのハッシュ値を添え字にして配列の要素を格納
# 上記の処理を配列の要素数分繰り返す

# 探索処理の速さを維持しつつも、メモリ使用量ができるだけ少ない2つのバランスが取れている
# 1.5倍〜2倍

# ハッシュリストの要素をすべて0で初期化


# ハッシュテーブルをチェイニングにし重複データを管理する O(N)
def hash_store(array: list) -> list:
  len_h = ceil(len(array) * 1.5)
  hash_table = [[] for _ in range(len_h)] # 各スロットにリストを初期化 [[data, index番号],[],[],[]...]
  for i, data in enumerate(array):
    k = data % len_h # ハッシュ値の計算
    # 同じハッシュ値を持つデータを格納する
    hash_table[k].append((data, i))
  return hash_table # (値, 元のインデックス)を格納

  # ハッシュテーブルからxに対応するインデックスを取得 O(1)
def hash_search(hash_table: list, x: int) -> list:
  k = x % len(hash_table) # ハッシュ値の計算
  # スロットのリストからxに対応するインデックスを取得
  result = [index for value, index in hash_table[k] if value == x]
  return result if result else None

x = 12
array_d = [12 ,25, 36, 25, 20, 30, 8, 25, 42]
hash_table = hash_store(array_d)
print('ハッシュテーブル：', hash_table)
hash_data = hash_search(hash_table, x)

if hash_data is None:
  print(f'{x}は存在しません')
else:
  print(f'格納場所は{hash_data}番目の要素です') 


array_n = [12, 25, 36, 20, 30, 8, 42]

'''
ハッシュ探索法（オープンアドレス法）

ハッシュスロット * 1.5個分用意したver
ハッシュ値がバッティングした場合は空き要素へ移動する
'''
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
選択ソート（整列）
時間計算量：O(N^2)

2重ループ構造

データを昇順や降順に並べ替えるアルゴリズム
最小値または最大値を探しだし先頭から順に並び替える方法

探索範囲の最小値を探す処理
探索範囲の最小値の先頭要素を交換する処理

最小値の添え字を記録していく

暫定の最小値の添え字i = 0を設定  
比較するための添え字を i + 1に設定
次の要素が暫定の最小値よりも小さい場合に最小値の添え字を更新する
ループ回数の制御

先頭のデータと最小値の格納されたデータを入れ替える

以上の処理を繰り返す

変数 i, kは動的に値が変化していく為、都度配列の参照する値も変わる
'''
array = [12, 13, 11, 14, 10] # 期待値：{10, 11, 12, 13, 14}

def select_sort(array: list) -> list:
  i = 0
  while i < len(array) - 1: # 4回回る（0-3）外側のループ
    index_min = i # 暫定の最小値の添え字を格納する
    k = i + 1
    while k < len(array): # 5回回る(0-4) 内側のループ
      # 配列要素が暫定最小値よりも小さければ
      if array[k] < array[index_min]:
        index_min = k # 最小値の添え字を更新する
      k += 1
    # 入れ替える処理
    w = array[i]
    array[i] = array[index_min]
    array[index_min] = w
    i += 1
  return array

print(select_sort(array))


'''
バブルソート

時間計算量：O(N^2)
n(n - 1) / 2回

2重ループ構造

例）
n = 4
4 * 3 / 2 = 6回
右端から順に隣り合ったデータ昇順に並べ替える
左端の枠から順に、入るデータを確定させていく

i = len(array) - 1
i・・・右の要素の添え字
i - 1・・・左の要素の添え字
k・・・並べ替えが確定した要素の回数

右端の要素＝array[i]
手前の要素＝array[i - 1]

隣り合ったデータの要素を比較する処理
逆順の場合入れ替える処理を繰り返す方法
昇順に並べ替える処理＝位置を入れ替える
1つ左の要素に移動する処理＝ i = i - 1
並べ替えた回数を保持する処理
'''

# 昇順に並べ変える
def bubble_sort(array: list) -> list:
  k, length = 0, len(array) - 1
  # すべての要素が整列するまで繰り返す
  while k < length:
    i = length
    while i > k:
      # 左の要素が昇順ではない時のみ入れ替える
      if array[i - 1] > array[i]:
        w = array[i - 1]
        array[i - 1] = array[i]
        array[i] = w
      i -= 1 # 参照する要素を左にさ
    k += 1
  return array

array = [5, 3, 4, 1, 2] # 期待値：[1, 2, 3, 4, 5]未整列
array2 = [1, 2, 3, 5, 4] # 期待値：[1, 2, 3, 4, 5]整列済み
array3 = [5, 3, 4, 1, 2, 10, 11, 14, 17, 20] # 期待値：[1, 2, 3, 4, 5]半分整列ずみ

print(bubble_sort(array))

'''
挿入ソート（整列）
時間計算量：O(N^2)

昇順または降順に並べ替える
整列済みと未整列と分ける考え方

0番目の配列を整列済みとみなす
1番目の以降の配列と0番目の要素を比較していく

挿入するデータ＝array[i]を準備
k ＝ 配列の空き要素の添え字
'''
# 挿入ソート（データを昇順に並べ替える）
def insert_sort(array: list) -> list:
  # 配列の件数分繰り返す
  i = 1
  while i < len(array):
    x = array[i] # 挿入するデータ
    k = i # 空き要素の添え字の管理
    # 昇順ではない場合のみ、一つ前の要素を空き要素にする
    while k > 0 and array[k - 1] > x:
      array[k] = array[k - 1]
      k -= 1
    # 直前に更新した空き要素枠へデータを挿入する
    array[k] = x
    i += 1
  return array

print(insert_sort(array))
print(insert_sort(array2))
print(insert_sort(array3))



data = [5, 4, 7, 6, 8, 3, 1, 2, 9]

# 基準値を決めてデータを大小のグループに分け昇順に並べ替える
def quick_sort(array: list) -> list:
  # 再帰処理の終了条件
  if len(array) <= 1:
    return array
  
  pivot = array[len(array) // 2] # 基準値の計算 8
  left  = [x for x in array if x < pivot] # 基準値よりも小さい要素を抽出
  middle= [x for x in array if x == pivot]# 基準値と等しい要素を抽出
  right = [x for x in array if x > pivot] # 基準値よりも大きい要素を抽出

  # 分割統治法（大きな問題を小さな問題に分割して、それぞれを独立して解くこと）
  return quick_sort(left) + middle + quick_sort(right)

print(quick_sort(data))

'''
クイックソート（難しい）
時間計算量：O(N log N)

基準値を決めてデータを大小のグループに分ける

array[left] = 基準値

left：並べ替え範囲の添え字を入れる変数
right：並べ替え範囲の末尾要素の添え字を入れる変数
i：基準値よりも大きい要素を探すための変数
k：基準値よりも小さい要素を探すための変数
w：交換用の変数

変数 i を使って基準値よりも大きい要素を探す
変数 k を使って基準値よりも小さい要素を探す

# ----再帰処理------
# k - 1＝確定した真ん中の要素を引いた数（2回目以降のクイックソートする範囲）

'''
# 基準値を決めてデータを大小に分けて昇順に並べ替える
def quick_sort_sub(array: list, left: int, right: int) -> list:
  i, k, w = left + 1, right, 0
  # 先頭と末尾がすれ違うまで繰り返す
  while i < k:
    # 基準値より大きい要素を右端まで探す処理
    while array[i] < array[left] and i < right:
      i += 1 # 順番に右へ移動していく

    #基準値より小さい要素を探す処理
    while array[k] > array[left] and k > left:
      k -= 1 # 順番に左に移動していく

    # 先頭と末尾がすれ違ってない場合のみ大小のデータを入れ替える
    if i < k:
      w = array[i]
      array[i] = array[k]
      array[k] = w
    
  # 基準値を大小の真ん中のデータに移動する
  if array[left] > array[k]:
    w = array[left]
    array[left] = array[k]
    array[k] = w
  
  # 再帰処理
  # 大小に分けたデータの範囲を再度クイックソートする
  if left < k - 1:
    quick_sort_sub(array, left, k - 1)
  if k + 1 < right:
    quick_sort_sub(array, k + 1, right)

  return array

print(quick_sort_sub(data, 0, len(data) - 1))