nums=[19,17]
j=0
for num in nums:
    for i in range(num):
        print(f'/{j}/'+f"00{i}.png {j}"if i<10 else f'/{j}/'+f"0{i}.png {j}")
    j+=1

# nums=[35,32,26,29]
# j=0
# for num in nums:
#     for i in range(num):
#         print(f'/{j}/'+f"00{i}.png {j}"if i<10 else f'/{j}/'+f"0{i}.png {j}")
#     j+=1