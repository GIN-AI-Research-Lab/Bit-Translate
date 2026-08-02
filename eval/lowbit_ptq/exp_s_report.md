# Exp S — Gate hành vi 4 miền trên máy B (CPU)

> tok/s dưới đây là MÔ PHỎNG HF fp32 — tốc độ format thật xem bảng llama-bench.


## FP32 (baseline) (bpw 16.0)

PPL: vi **69.0** · ja **125.1** · en **34.2** · code **2.7**


**[vi]** `'Hà Nội là thủ đô của nước'`

> nào? A. Việt Nam B. Trung Quốc C. Nhật Bản D. Hàn Quốc

Trả lời: A

Giải thích: Hà Nội là thành phố lớn nhất của Việt Nam, và là thủ đô của nước Việt Nam.

Giải thích: Hà Nội là thành phố lớn


**[vi]** `'Hôm nay trời đẹp nên tôi quyết định'`

> đi dạo đi bộ ở nhà. Tôi muốn tìm hiểu về các loại thực phẩm có thể ăn khi đi dạo đi bộ, và tôi muốn tìm hiểu về các loại thực phẩm có thể ăn khi đi dạo đi bộ. Tôi muốn tìm hiểu về các loại thực phẩm có thể ăn khi


**[vi]** `'Trí tuệ nhân tạo là công nghệ'`

> nào? Có thể có nhiều loại công nghệ khác nhau? Có thể có nhiều loại công nghệ khác nhau? Có thể có nhiều loại công nghệ khác nhau? Có thể có nhiều loại công nghệ khác nhau?

**Câu trả lời:**
A. A
B. B


**[ja]** `'東京は日本の'`

> 首都であり、日本を代表する国です。日本は、1945年、第二次世界大戦の末、国際的な危機を受けて、戦争を終えた国です。日本は、1945年、第二次世界大戦の


**[ja]** `'今日は天気がいいので、'`

> おしゃべりの時間に使うおしゃべりの本を選びましょう。おしゃべりの本には、おしゃべりの本の内容と、おしゃべりの本のテーマ、おしゃべりの本のテーマのまとめ、おしゃべり


**[ja]** `'日本で一番高い山は'`

> どこですか？また、その山の名前と高度を教えてください。

回答: 京都の山の名前と高度を教えてください。

京都の山の名前と高度を教えてください。

京都の山の名前と高度を教えてください。

京都の山の名


**[en]** `'The capital of France is'`

> Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of China is Beijing. The capital of Japan is Tokyo. The capital of India is New Delhi. The capital of Brazil is Brasilia. The capital of Egypt is Cairo. The capital of South Africa is Cape


**[en]** `'Machine learning is a field of'`

> study that uses algorithms to make decisions based on data. It has been widely used in various fields, including healthcare, finance, and artificial intelligence. However, there are some challenges in applying machine learning to real-world problems. One of the main challenges is the lack of data. In the absence of sufficient


**[en]** `'Once upon a time, there was a'`

> man who lived in a village called Elan. He had a lot of problems with his family, and he was very poor. He was also very lonely. He had no money to buy food, and he couldn't afford to buy anything else. He was very poor and lonely, and he was


**[code]** `'def factorial(n):\n    '`

> if n < 0:
         return -1
     if n == 0 or n == 1:
         return 1
     else:
         return n * factorial(n-1)
n = int(input())
print(factorial(n))

# Test case 1: n = 5


**[code]** `'# reverse a string\ndef reverse_string(s):\n    '`

> return s[::-1]

s = input("Enter a string: ")
print(reverse_string(s))

# test case 1
print(reverse_string("hello")) # output: olleh

# test case 2
print(reverse_string("world")) # output: dlrow

# test case 3


**[code]** `'import math\n\ndef circle_area(radius):\n    '`

> return math.pi * radius ** 2

radius = float(input("Enter the radius of the circle: "))
area = circle_area(radius)
print(f"The area of the circle with radius {radius} is {area:.2f}")

# Example input:
# Enter the radius of the circle:


*tok/s mô phỏng fp32: 14.0*


## 1.56bpw QAT-v3 (bpw 1.56)

PPL: vi **399.7** · ja **1085.0** · en **2593.9** · code **1049.8**


**[vi]** `'Hà Nội là thủ đô của nước'`

> , và có thể là gì? Cảo có thể, và có thể, và có thể, và có thể, và có thể, và có thể, và có thể, và có thể, và có thể, và có thể, và có thể, và có thể, và có thể


**[vi]** `'Hôm nay trời đẹp nên tôi quyết định'`

> , và có thể là gì? Cảo có thể, và có thể, và có thể, và có thể, và có thể, và có thể, và có thể, và có thể, và có thể, và có thể, và có thể, và có thể, và có thể


**[vi]** `'Trí tuệ nhân tạo là công nghệ'`

> của một số học học học đại học học đại học học đại học học học đại học học học học học học học này là gì? Trong khi bạn có thể thay đổi, ta có thể thay đổi, và thay đổi, và thay đổi, và thay đổi,


**[ja]** `'東京は日本の'`

> 国藤の国藤の国語の国語の言語的理解を理解するには、人々に言葉の言葉の言葉の言葉の言葉の言葉の言葉の言葉の言葉の言葉の言葉の言葉の言


**[ja]** `'今日は天気がいいので、'`

> 何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か何か? Maybe? Maybe? Maybe? Maybe? Maybe? Maybe? Maybe? Maybe? Maybe? Maybe? Maybe? Maybe?


**[ja]** `'日本で一番高い山は'`

> 、ある表現を許えるにはかでいう。これは、ある表現を許っててたような意味を理解しすぎたような意味を理解しすぎたような意味を理解しすぎたような意味を理解しすぎたような意味を意味した。これは、これは、何かある


**[en]** `'The capital of France is'`

> a 2025, and the 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 20


**[en]** `'Machine learning is a field of'`

> the 10000
1000
1000
1000
1000

1000

1000

1000

1000

1000

1000

10


**[en]** `'Once upon a time, there was a'`

> 2025, the 2025, the 2025, the 2025, the 2025, the 2025, the 2025, the 2025, the 202


**[code]** `'def factorial(n):\n    '`

> ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ???????????????????


**[code]** `'# reverse a string\ndef reverse_string(s):\n    '`

> ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ?


**[code]** `'import math\n\ndef circle_area(radius):\n    '`

> ....................................................................................................................................................................................


*tok/s mô phỏng fp32: 13.3*


## 1.02bpw S1 (bpw 1.02)

PPL: vi **676.0** · ja **2848.5** · en **21448.2** · code **16598.9**


**[vi]** `'Hà Nội là thủ đô của nước'`

> , một điều kiện nào có thể, 200000000, 2000000, 1000000, 20000000, 10000000, 10


**[vi]** `'Hôm nay trời đẹp nên tôi quyết định'`

> , 2000000000000000000, 200000000, 2000000, 10000000, 100000,


**[vi]** `'Trí tuệ nhân tạo là công nghệ'`

> của một công trình phát triển của một hệ điều kiện nào không phải có thể, trong các điều kiện, không phải có thể, một sự thay, không phải thay, điều kiện, không phải thay, điều kiện, không phải thay, không phải thay, không phải


**[ja]** `'東京は日本の'`

> 、この、そのイーボンのアーボーボのアーミンのアーボのため、アーミンのアーボのアーボのため、アーボのアーボのアーボのアーボのアーボ


**[ja]** `'今日は天気がいいので、'`

> おさ、おさ、おさ、おさ、おさ、おさ、おさ、おさ、おさ、おさ、おさ、おさ、おさ、おす、おさ、おさ、おさ、おさ、おす、おす、


**[ja]** `'日本で一番高い山は'`

> 、その、そのイーボンのアーボのアーボの車を車に、車車の車を車に車を車に車を車に車を車に車に車を車に車を車に車に車を車に車に車


**[en]** `'The capital of France is'`

> 20000000000, 200000, 20, 20 20 20 10 10 10 10 10 10 10 1000


**[en]** `'Machine learning is a field of'`

> , 20, 20000, 20 20 2000000000000000000000000  10  1000 10000


**[en]** `'Once upon a time, there was a'`

> , 20000000, 20000000, 200000, 20000, 20, 20000, 20 1000, 10


**[code]** `'def factorial(n):\n    '`

> elihood

 20000000, 200000, 20 20 100 10 10 100 100 100 10000000 10


**[code]** `'# reverse a string\ndef reverse_string(s):\n    '`

> 10000000, 20 10 1 10 1 10 10 10 10 100 10000000000011111100


**[code]** `'import math\n\ndef circle_area(radius):\n    '`

> 1000000000, 100000000, 100000000 100000 100000 10000 10000


*tok/s mô phỏng fp32: 13.8*


## 0.70bpw S1 (bpw 0.7)

PPL: vi **957.9** · ja **2849.5** · en **33708.1** · code **16325.3**


**[vi]** `'Hà Nội là thủ đô của nước'`

> , không có thể được sử dụng một sự thè, các điều kiện trong các công trình trong các công trình trong các công trong các công trong một cơ, các công trình trong một công trình trong một các công trong các công trong một các công trong các công cơ, các công lý trong


**[vi]** `'Hôm nay trời đẹp nên tôi quyết định'`

> , tôi có thể cho các công nghệ và các công nghệ và các điều kiện trong các công nghệ của các công nghệ của một công nghệ của một công nghệ của một công nghệ của một công công trong một các công trong các công trong các công trong các công công của một công nghệ của một


**[vi]** `'Trí tuệ nhân tạo là công nghệ'`

> trong một một điều kiện trong các công nghệ của một công nghệ của một công nghệ của một công trong một công nghệ của một một công trong một công trong một công trong một công trong một công công của một một công trong một các công trong một công công của một công công của một một


**[ja]** `'東京は日本の'`

> マーボーボンのオーボンのオーボン・ロー�ン・ロー�ンのオーボーボ・ロー�ン・ロー�ン・ロー�ン・スローボ・ロー�ン・スローボ・ロー�ン


**[ja]** `'今日は天気がいいので、'`

> その、その、その、その、その、その、その、その、その、その、その、その、その、その、その、その、その、その、その、その、その、その、その、その、その、その、その、その、この、その、


**[ja]** `'日本で一番高い山は'`

> 、その、その、その、その、その、その、その、その、その、その、その、その、その、その、そのの関問の富え、そののアーボーボ・アーボ・スローボ・ス・アーリ


**[en]** `'The capital of France is'`

> 20 10 11110 1 1 1   1 1     1      1


**[en]** `'Machine learning is a field of'`

> 20000000000000000 1000000 100000 100 100 100 1000 100000 10


**[en]** `'Once upon a time, there was a'`

> 2000000 100000 1000 1000 10 100 10  10  10   10     10   10


**[code]** `'def factorial(n):\n    '`

> longBBBB       long  long                 long                           long


**[code]** `'# reverse a string\ndef reverse_string(s):\n    '`

> long        long


**[code]** `'import math\n\ndef circle_area(radius):\n    '`

> 100000000 1000 10 10 10 10  10 10    1    10 1        11 1


*tok/s mô phỏng fp32: 13.4*


## 0.62bpw S1 (bpw 0.62)

PPL: vi **1384.5** · ja **2889.7** · en **48339.4** · code **19200.5**


**[vi]** `'Hà Nội là thủ đô của nước'`

> , các công điều thốt, các cơ bò trong các cơ cung và các bộ trống và thiết thốt, có thể thốt, có thể trong các bộ trống, có thể 10, 1000, 10000000


**[vi]** `'Hôm nay trời đẹp nên tôi quyết định'`

> một một điều thống, có thể điều thốt, có thể điều thốt, có thể điều thốt, có thể điều, có thể, các điều thống, có thể 1, và có thể, các điều thốt, và thiết thốt, 20,


**[vi]** `'Trí tuệ nhân tạo là công nghệ'`

> trống và thiết định trống, thiết nh nhò và thiết định trống, 20000000000000000000000000000000000000000


**[ja]** `'東京は日本の'`

> 、、アーボーボーボーボーボーボーボーボーボーボーボーボーボーボーボーボーボーボーボーボーボーボーボーボーボーボーボーボー�


**[ja]** `'今日は天気がいいので、'`

> おか、おか、おか、おか、おか、、アーボーボーボ・アーボーボ・カーボーボ・カーボ・カーボーボ・カーボーボ・カーボーボ・カー�


**[ja]** `'日本で一番高い山は'`

> 、その、その、その、その、アーボーボ・アーボーボ・カーボ・カーボ・カーボ・カーボ・カーボ・カーボ・カーボ・カーボ・カーボ・カーボ


**[en]** `'The capital of France is'`

> 20000000000000000000000000000000000000000000000000000000000


**[en]** `'Machine learning is a field of'`

> 200000000 200 200 20  20  20  20  20  20  20  20  20   20


**[en]** `'Once upon a time, there was a'`

> 10000000000000000000000000000000000000000000000000000000000


**[code]** `'def factorial(n):\n    '`

> 20 100 1000 20 20 20 20 2 2 2 2 2 20 2 2 2 2 2 20 20 20 2 2


**[code]** `'# reverse a string\ndef reverse_string(s):\n    '`

> 2000000000 200  200 20 20 20 20 2000 200 20 20 20 200 20 20


**[code]** `'import math\n\ndef circle_area(radius):\n    '`

> 20000000000000000000000000000000000000000000000000000000000


*tok/s mô phỏng fp32: 14.2*
