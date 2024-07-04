



# # Check prediction
# # ROI = check_ROI(array_3d)
# # print(ROI)

# # Fit the model for our chosen sample.
# for i in range(array_3d.shape[0]):
#     # the prediction
#     array_3d[:,:,6] = check_prediction(array_3d[i,:,:], model)

# # Now we want to graph the stock price during this time and add a green dot when we buy and a red dot when we close.

# import matplotlib.pyplot as plt

# # Get the stock prices
# stock_prices = array_3d[:, 0, 0]

# # Create the plot
# plt.plot(stock_prices)

# # Add green dots for buy positions
# buy_positions = np.where(array_3d[:, 0, 6] == 'buy')[0]
# plt.scatter(buy_positions, stock_prices[buy_positions], color='green', marker='o')

# # Add red dots for sell positions
# sell_positions = np.where(array_3d[:, 0, 6] == 'close')[0]
# plt.scatter(sell_positions, stock_prices[sell_positions], color='red', marker='o')

# # Show the plot
# plt.show()


        



