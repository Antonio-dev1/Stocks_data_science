import tkinter
import tkinter.messagebox
import customtkinter
from SentimentAnalysis import runStockPredictionSentiment, runStockPredictionSentimentWithSavedModel
from Stock_Prediction_No_Sentiment import runPredictionWithoutSentiment, runPredictionWithoutSentimentWithSavedModels
from Stock_Prediction_LSTM import runLSTM, runLSTMWithSavedModel
from Crypto_Analysis import runCryptoAnalysis, runCryptoAnalysisWithSavedModels
from Crypto_LSTM import runCryptoLSTM, runCryptoLSTMWithSavedModel

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
import pickle

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Buy High, Sell Low")
        self.geometry(f"{1200}x{580}")
        self.isDrawn = False

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets

        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Asset Manager",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=self.onCryptoPressed,
                                                        text="Crypto")
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, command=self.regularStockPressed,
                                                        text="Stock_No_Sentiment")
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, command=self.stock_sentimentClicked,
                                                        text="Stock+Sentiment")
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.checkbox_var = tkinter.IntVar(value=1)
        self.checkbox_getSignal = customtkinter.CTkCheckBox(master=self.sidebar_frame, text="getSignal", height=10,
                                                            variable=self.checkbox_var)
        self.checkbox_getSignal.grid(row=4, column=0, pady=(20, 0), padx=20, sticky="n")
        # set default values
        self.checkbox_getSignal.select()
        self.appearance_mode_optionemenu.set("Dark")

        # create main entry and button
        self.model_output = customtkinter.CTkLabel(self, text='')
        self.model_output.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        # create radiobutton frame
        self.radiobutton_frame = customtkinter.CTkFrame(self)

        self.radiobutton_frame.grid(row=1, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.radio_var = tkinter.IntVar(value=0)
        self.radio_var_saved = tkinter.IntVar(value=1)
        self.label_radio_group = customtkinter.CTkLabel(master=self.radiobutton_frame, text="Models:")
        self.label_radio_group.grid(row=1, column=0, columnspan=1, padx=10, pady=10, sticky="")
        self.radio_button_1 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var,
                                                           text="LinearRegression", value=0)
        self.radio_button_1.grid(row=1, column=3, pady=10, padx=20, sticky="n")
        self.radio_button_2 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var,
                                                           text="Ridge", value=1)
        self.radio_button_2.grid(row=2, column=3, pady=10, padx=20, sticky="n")
        self.radio_button_3 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var,
                                                           text="Lasso", value=2)
        self.radio_button_3.grid(row=3, column=3, pady=10, padx=20, sticky="n")
        self.radio_button_4 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var,
                                                           text="LSTM", value=3)
        self.radio_button_4.grid(row=4, column=3, pady=10, padx=20, sticky="n")
        self.radio_button_5 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var,
                                                           text="SVR", value=4)
        self.radio_button_5.grid(row=5, column=3, pady=10, padx=20, sticky="n")
        self.radio_button_6 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var,
                                                           text="XGB", value=5)
        self.radio_button_6.grid(row=6, column=3, pady=10, padx=20, sticky="n")

        self.checkbox_var_models = tkinter.IntVar(value=1)
        self.checkbox_use_saved = customtkinter.CTkCheckBox(master=self.radiobutton_frame, text="Use Saved Models",
                                                            height=10,
                                                            variable=self.checkbox_var_models)
        self.checkbox_use_saved.grid(row=7, column=3, pady=10, padx=20, sticky="n")
        # set default values
        self.checkbox_use_saved.select()

        # Create a frame to display the graph

        self.frame = customtkinter.CTkFrame(self, height=1100)
        self.frame.grid(row=1, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")
        self.figure = plt.figure(figsize=(10, 10), dpi=100)

        # create a canvas to display the figure
        self.canvas = tkagg.FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.get_tk_widget().pack()

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("sidebar_button click")

    def stock_sentimentClicked(self):
        plt.clf()

        ax = self.figure.add_subplot(111)
        if self.isDrawn:
            ax.clear()
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            print('canva cleared')
            self.canvas.draw()

        modelsMap = {0: 'LinearRegression', 1: 'Ridge', 2: 'Lasso', 3: 'LSTM', 4: "SVR", 5: "XGB"}
        modelName = modelsMap[self.radio_var.get()]
        if self.checkbox_getSignal.get() == 1:
            if self.checkbox_use_saved.get() == 1:
                print(modelName)
                predicted_prices, real_prices, output_df, frames, buyingsignals, sellingdates, winning_rate = runStockPredictionSentimentWithSavedModel(
                    modelName, self.checkbox_getSignal.get())
            else:
                predicted_prices, real_prices, output_df, frames, buyingsignals, sellingdates, winning_rate = runStockPredictionSentiment(
                    modelName, self.checkbox_getSignal.get())
            ax = self.figure.add_subplot(111)
            ax.scatter(frames.loc[buyingsignals].index, frames.loc[buyingsignals]['Adj Close'], marker='^', c='g')
            ax.plot(frames['Adj Close'], alpha=0.7)
            ax.scatter(frames.loc[sellingdates].index, frames.loc[sellingdates]['Adj Close'], marker='^', c='r')
            ax.plot(frames['Adj Close'], alpha=0.7)
            ax.set_title('Buying and Selling Signals for the predictions done by ' + str(modelName) + " model")
            self.model_output.configure(text="Winning rate of strategy is " + str(winning_rate))
            self.canvas.draw()

        elif self.checkbox_getSignal.get() == 0:
            if self.checkbox_use_saved.get() == 1:
                print(modelName)
                predicted_prices, real_prices, output_df, frames, buyingsignals, sellingdates, winning_rate = runStockPredictionSentimentWithSavedModel(
                    modelName, self.checkbox_getSignal.get())
            else:
                predicted_prices, real_prices, output_df, RMSE, Rsquared = runStockPredictionSentiment(modelName,
                                                                                                   self.checkbox_getSignal.get())
            ax = self.figure.add_subplot(111)
            ax.plot(output_df.index, output_df['Real'], c='y')
            ax.plot(output_df.index, output_df['Adj Close'], c='orange')
            ax.legend(['Actual', 'Predicted'])
            ax.set_title('This is the graph for the prediction of the ' + modelName + " model")
            self.model_output.configure(text="R-squared " + str(Rsquared) + "\n" + "RMSE: " + str(RMSE))
            self.canvas.draw()
            self.isDrawn = True

    def regularStockPressed(self):
        plt.clf()

        ax = self.figure.add_subplot(111)
        if self.isDrawn:
            ax.clear()
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            print('canva cleared')
            self.canvas.draw()

        modelsMap = {0: 'LinearRegression', 1: 'Ridge', 2: 'Lasso', 3: 'LSTM', 4: "SVR", 5: "XGB"}
        modelName = modelsMap[self.radio_var.get()]

        if modelName == 'LSTM':
            plotLSTM(self, self.checkbox_getSignal.get())
        else:
            if self.checkbox_getSignal.get() == 1:
                plt.clf()
                if self.checkbox_use_saved.get() == 1:
                    print(modelName)
                    predicted_prices, real_prices, output_df, frames, buyingsignals, sellingdates, winning_rate = runPredictionWithoutSentimentWithSavedModels(
                        modelName, self.checkbox_getSignal.get())
                else:
                    predicted_prices, real_prices, output_df, frames, buyingsignals, sellingdates, winning_rate = runPredictionWithoutSentiment(
                    modelName, self.checkbox_getSignal.get())
                ax = self.figure.add_subplot(111)
                ax.scatter(frames.loc[buyingsignals].index, frames.loc[buyingsignals]['Adj Close'], marker='^', c='g')
                ax.plot(frames['Adj Close'], alpha=0.7)
                ax.scatter(frames.loc[sellingdates].index, frames.loc[sellingdates]['Adj Close'], marker='^', c='r')
                ax.plot(frames['Adj Close'], alpha=0.7)
                ax.set_title('Buying and Selling Signals for the predictions done by ' + str(modelName) + " model")
                self.model_output.configure(text="Winning rate of strategy is " + str(winning_rate))
                self.canvas.draw()

                self.isDrawn = True


            elif self.checkbox_getSignal.get() == 0:
                plt.clf()
                if self.checkbox_use_saved.get() == 1:
                    print(modelName)
                    predicted_prices, real_prices, output_df, frames, buyingsignals, sellingdates, winning_rate = runPredictionWithoutSentimentWithSavedModels(
                        modelName, self.checkbox_getSignal.get())
                else:
                    predicted_prices, real_prices, output_df, RMSE, Rsquared = runPredictionWithoutSentiment(modelName,
                                                                                                         self.checkbox_getSignal.get())
                ax = self.figure.add_subplot(111)
                ax.plot(output_df.index, output_df['Real'], c='y')
                ax.plot(output_df.index, output_df['Adj Close'], c='orange')
                ax.legend(['Actual', 'Predicted'])
                ax.set_title('This is the graph for the prediction of the ' + modelName + " model")
                self.model_output.configure(text="RMSE: " + str(RMSE))
                self.canvas.draw()

                self.isDrawn = True

    def onCryptoPressed(self):
        plt.clf()

        ax = self.figure.add_subplot(111)
        if self.isDrawn:
            ax.clear()
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            print('canva cleared')
            self.canvas.draw()

        modelsMap = {0: 'LinearRegression', 1: 'Ridge', 2: 'Lasso', 3: 'LSTM', 4: "SVR", 5: "XGB"}
        modelName = modelsMap[self.radio_var.get()]
        if modelName == 'LSTM':
            plotLSTMCrypto(self, self.checkbox_getSignal.get())
        else:
            if self.checkbox_getSignal.get() == 1:
                plt.clf()
                if self.checkbox_use_saved.get() == 1:
                    print(modelName)
                    preds, test_data, window_size, buy_signals, sell_signals = runCryptoAnalysisWithSavedModels(
                        self.checkbox_getSignal.get(), modelName)
                else:
                    preds, test_data, window_size, buy_signals, sell_signals = runCryptoAnalysis(
                    self.checkbox_getSignal.get(), modelName)
                ax = self.figure.add_subplot(111)
                ax.plot(test_data.index[window_size:], preds, label='Predicted', linewidth=2)
                ax.scatter(test_data.index[window_size:][buy_signals], preds[buy_signals], color='green',
                           label='Buy Signal',
                           marker='^', zorder=3)
                ax.scatter(test_data.index[window_size:][sell_signals], preds[sell_signals], color='red',
                           label='Sell Signal',
                           marker='v', zorder=3)
                self.canvas.draw()

                self.isDrawn = True

            elif self.checkbox_getSignal.get() == 0:
                plt.clf()
                if self.checkbox_use_saved.get() == 1:
                    print(modelName)
                    preds, test_data, window_size, buy_signals, sell_signals = runCryptoAnalysisWithSavedModels(
                        self.checkbox_getSignal.get(), modelName)
                else:
                    preds, test_data, train_data, y_test, window_size, mae, mse, r2, data, X_test = runCryptoAnalysis(
                        self.checkbox_getSignal.get(),
                            modelName)
                train_size = len(train_data)
                ax = self.figure.add_subplot(111)
                ax.plot(data.index[window_size + train_size:], data['Close'][window_size + train_size:],
                        label='Actual',
                        linewidth=2)
                ax.plot(test_data.index[window_size:], preds, label='Predicted', linewidth=2)
                ax.set_xlabel('Date')
                ax.set_ylabel('BTC/USD')
                ax.set_title('BTC/USD Prediction using ' + str(modelName) + ' with Rolling Window')
                self.model_output.configure(
                    text="MAE: " + str(mae) + "\n" + "MSE: " + str(mse) + "\n" + "R2: " + str(r2))
                self.canvas.draw()
                self.isDrawn = True



def plotLSTM(self, getSignal):
    plt.clf()

    ax = self.figure.add_subplot(111)
    if self.isDrawn:
        ax.clear()
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        print('canva cleared')
        self.canvas.draw()

    if getSignal:
        if self.checkbox_use_saved.get() == 1:
            print("Regular LSTM")
            predicted_prices, real_prices, output_df, frames, buyingsignals, sellingdates, winning_rate = runLSTMWithSavedModel(
                getSignal)
        else:
            predicted_prices, real_prices, output_df, frames, buyingsignals, sellingdates, winning_rate = runLSTM(getSignal)
        ax = self.figure.add_subplot(111)
        ax.scatter(frames.loc[buyingsignals].index, frames.loc[buyingsignals]['Adj Close'], marker='^', c='g')
        ax.plot(frames['Adj Close'], alpha=0.7)
        ax.scatter(frames.loc[sellingdates].index, frames.loc[sellingdates]['Adj Close'], marker='^', c='r')
        ax.plot(frames['Adj Close'], alpha=0.7)
        ax.set_title('Buying and Selling Signals for the predictions done by LSTM' + " model")
        self.model_output.configure(text="Winning rate of strategy is " + str(winning_rate))
        self.canvas.draw()
        self.isDrawn = True


    else:
        if self.checkbox_use_saved.get() == 1:
            print("Regular LSTM")
            predicted_prices, real_prices, output_df, frames, buyingsignals, sellingdates, winning_rate = runLSTMWithSavedModel(
                getSignal)
        else:
            predicted_prices, real_prices, output_df, RMSE = runLSTM(getSignal)
        ax = self.figure.add_subplot(111)
        ax.plot(output_df.index, output_df['Real'], c='y')
        ax.plot(output_df.index, output_df['Adj Close'], c='orange')
        ax.legend(['Actual', 'Predicted'])
        ax.set_title("This is the graph for the prediction of the LSTM model")
        self.model_output.configure(text="RMSE: " + str(RMSE))
        self.canvas.draw()
        self.isDrawn = True


def plotLSTMCrypto(self , getSignal):
    plt.clf()

    ax = self.figure.add_subplot(111)
    if self.isDrawn:
        ax.clear()
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        print('canva cleared')
        self.canvas.draw()

    if getSignal:
        if self.checkbox_use_saved.get() == 1:
            print("Crypto LSTM")
            preds, test_data, window_size, buy_signals, sell_signals = runCryptoLSTMWithSavedModel(getSignal)
        else:
            preds, test_data, window_size, buy_signals, sell_signals = runCryptoLSTM(getSignal)
        ax = self.figure.add_subplot(111)
        ax.plot(test_data.index[window_size:], preds, label='Predicted', linewidth=2)
        ax.scatter(test_data.index[window_size:][buy_signals], preds[buy_signals], color='green',
                   label='Buy Signal',
                   marker='^', zorder=3)
        ax.scatter(test_data.index[window_size:][sell_signals], preds[sell_signals], color='red',
                   label='Sell Signal',
                   marker='v', zorder=3)
        self.canvas.draw()
        self.isDrawn = True


    else:
        if self.checkbox_use_saved.get() == 1:
            print("Crypto LSTM")
            preds, test_data, window_size, buy_signals, sell_signals = runCryptoLSTMWithSavedModel(getSignal)
        else:
            preds, targets, mse = runCryptoLSTM(getSignal)
        ax = self.figure.add_subplot(111)
        ax.plot(targets, label='actual', linewidth=2)
        ax.plot(preds, label='predictions', linewidth=2)
        ax.set_ylabel('BTC/USD', fontsize=14)
        ax.set_title('This is the model using LTSM', fontsize=16)
        ax.legend(loc='best', fontsize=16)
        self.model_output.configure(text="MSE: " + str(mse))
        self.canvas.draw()
        self.isDrawn = True


if __name__ == "__main__":
    app = App()
    app.mainloop()
