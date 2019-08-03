package sample;

import javafx.application.Application;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXMLLoader;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.control.Button;
import javafx.scene.Scene;
import javafx.scene.control.TextField;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;


import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.Random;


public class Main extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception{
//        Parent root = FXMLLoader.load(getClass().getResource("sample.fxml"));
        primaryStage.setTitle("Strong Password Generator");


        // Button Node
        final Button button = new Button("Generate!");
        button.setMaxSize(150, 50);


        // TextField Node
        final TextField textField = new TextField();
        textField.setMaxSize(150, 20);


        // Pane Init
        GridPane gridPane = new GridPane();

        gridPane.setHgap(10);
        gridPane.setVgap(10);
        gridPane.setPadding(new Insets(10, 10, 10, 10));
        gridPane.setAlignment(Pos.CENTER);
        gridPane.add(textField, 1, 0);
        gridPane.add(button, 1, 1);

        Scene mainScene = new Scene(gridPane, 500, 300);

        /*
        Press 'Q' on a keyboard mean Quit from app
         */
        mainScene.addEventFilter(KeyEvent.KEY_PRESSED,  new EventHandler<KeyEvent>(){
            public void handle(KeyEvent keyEvent)
            {
                if (keyEvent.getCode() == KeyCode.Q)
                {
                    primaryStage.close();
                }
            }
        });

        primaryStage.setScene(mainScene);
        primaryStage.show();


        // Button Action
        button.setOnAction(new EventHandler<ActionEvent>() {
            public void handle(ActionEvent event) {
                textField.setText(generate(16));
            }
        });

    }

    public static String generate(int passwordLength)
    {
        if (passwordLength < 8 || passwordLength > 28)
            throw new IllegalArgumentException("\nToo large or too small password.\nPlease try another length.");

        // Set of characters that will be used in a password
        String[] symbols = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                "#", "$", "!", "<", ">", "?", "/", "{", "}", "&", "*", "~", "+", "^", "%", "@", "[", "]"
        };



        try {

            Random random = SecureRandom.getInstanceStrong();
            StringBuilder sb = new StringBuilder(passwordLength);
            for (int i = 0; i < passwordLength; i++) {
                int indexRandom = random.nextInt( symbols.length );
                sb.append( symbols[indexRandom] );
            }
            return sb.toString();

        } catch (NoSuchAlgorithmException exc) {
            exc.printStackTrace();
        }

        return null;
    }


    public static void main(String[] args) {

        launch(args);



    }
}
