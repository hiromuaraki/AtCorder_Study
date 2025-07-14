'''
2025/3/13
最大公約数を求める
ユークリッドの互除法
わられる数 = わる数 × 商 + 余り
・ゼロになるまで余りを取っていく
・最後に割った数が最大公約数

再帰処理 y(割った数と余りが変化していくのが肝)
'''
def god(x, y):
  # 最大公約数を返す
  if x % y == 0:
    return y
  # 余りがゼロになるまで繰り返す
  return god(y, x % y)

x, y = 2431, 1496
print(f'{x},{y}の最大公約数＝{god(x, y)}')



'''
偶数の要素のみの合計・平均を計算するプログラム
その他リストの操作の練習
'''
scores = [88, 85, 90, 120, 45] # 点数を管理
total = 0
even_list = []
odd_list = [50, 60, 70]
# 点数を一つずつリストから取り出し合計する（合計するならtotal関数）
for score in scores:
  if score % 2 == 0:
    # 偶数だけ加算する
    total += score
    # リストへ要素の追加
    even_list.append(score)


odd_list.insert(0, 200)
odd_list[3] = 120 # indexを指定し要素上書き
print(f'奇数のリスト：{odd_list}')
print(f'リスト：{scores[::-1]}') # 逆順に表示
print(f'リスト：{scores[2:4]}') # 範囲表示 90, 120
print(f'{scores[1:]}') # 1番目以降を表示

print(f'要素の取り出し：{odd_list.pop()}')
print(f'要素数：{len(scores)}') # リストの要素数
print(f'偶数のリスト：{even_list}')
print(f'合計＝{total}')
print(f'平均＝{total // len(even_list)}')