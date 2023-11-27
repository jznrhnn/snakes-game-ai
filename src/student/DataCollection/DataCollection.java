package student.DataCollection;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import negasnake.NegaSnake;
import snakes.Bot;
import snakes.Coordinate;
import snakes.Direction;
import snakes.Snake;

public class DataCollection implements Bot {
    private static final String LOG_PATH = "DataCollection.log";
    private static final String DATA_PATH = "data//DataCollection.csv";

    // record apple generate time and location
    private long appleTime = System.currentTimeMillis();

    // record bot start time
    private long gameTime = System.currentTimeMillis();

    // info
    public Snake lastSnake;
    public Snake lastOpponent;
    public Coordinate lastApple = null;
    public Boolean forecast;
    public long lastGameTime;

    // last state
    List<Integer> lastState = new ArrayList<>();
    Direction lastDirection;

    public static int directionToNumber(Direction direction) {
        switch (direction) {
            case UP:
                return 0;
            case DOWN:
                return 1;
            case LEFT:
                return 2;
            case RIGHT:
                return 3;
            default:
                return -1;
        }
    }

    // Number to direction
    public static Direction numberToDirection(int number) {
        switch (number) {
            case 0:
                return Direction.UP;
            case 1:
                return Direction.DOWN;
            case 2:
                return Direction.LEFT;
            case 3:
                return Direction.RIGHT;
            default:
                return null;
        }
    }

    /**
     * Choose the direction (not rational - silly)
     * 
     * @param snake    Your snake's body with coordinates for each segment
     * @param opponent Opponent snake's body with coordinates for each segment
     * @param mazeSize Size of the board
     * @param apple    Coordinate of an apple
     * @return Direction of bot's move
     */
    @Override
    /* choose the direction (stupidly) */
    public Direction chooseDirection(Snake snake, Snake opponent, Coordinate mazeSize, Coordinate apple) {
        long startTime = System.currentTimeMillis();
        Direction resDirection = null;
        // log current snake and apple position, opponent snake position
        output("------------------" + new Date() + "------------------");
        output("Snake: " + snake.getHead());
        output("snake body" + snake.body);
        output("Opponent: " + opponent.getHead());
        output("Opponent body" + opponent.body);
        output("Apple: " + apple);
        output("Game time: " + (System.currentTimeMillis() - gameTime));

        // get apple disappear time
        if (lastApple != null && !lastApple.equals(apple)) {
            appleTime = System.currentTimeMillis();
        }
        long existTime = System.currentTimeMillis() - appleTime;
        output("apple exist time: " + existTime);

        List<Integer> state = generateOneHotState(snake, opponent, mazeSize, apple, existTime);

        // get dirction from negaSnake
        try {
            NegaSnake negaSnake = new NegaSnake();
            resDirection = negaSnake.chooseDirection(snake, opponent, mazeSize, apple);
        } catch (Exception e) {
            e.printStackTrace();
            output("NegaSnake error:" + e.getMessage());
        }

        // save state and reward on the file
        List<Integer> datas = new ArrayList<>();
        datas.addAll(state);
        datas.add(directionToNumber(resDirection));
        saveDataToCSV(datas, DATA_PATH);
        lastApple = apple;
        output(resDirection.toString());
        output("Time cost: " + (System.currentTimeMillis() - startTime) + "ms");
        return resDirection;
    }

    private List<Integer> generateOneHotState(Snake snake, Snake opponent, Coordinate mazeSize, Coordinate apple,
            long existTime) {
        // generate state
        List<Integer> state = new ArrayList<>();
        // snake state
        Integer[] snakeState = new Integer[mazeSize.x * mazeSize.y];
        Object[] bodys = snake.body.toArray();
        Arrays.fill(snakeState, 0);
        for (int i = 0; i < bodys.length; i++) {
            int index = ((Coordinate) bodys[i]).y * mazeSize.x + ((Coordinate) bodys[i]).x;
            snakeState[index] = 1;
        }
        Collections.addAll(state, snakeState);

        // oppnent state
        Integer[] opponentState = new Integer[mazeSize.x * mazeSize.y];
        Object[] opponentBodys = opponent.body.toArray();
        Arrays.fill(opponentState, 0);

        for (int i = 0; i < opponentBodys.length; i++) {
            int index = ((Coordinate) opponentBodys[i]).y * mazeSize.x + ((Coordinate) opponentBodys[i]).x;
            opponentState[index] = 1;
        }
        Collections.addAll(state, opponentState);

        // add apple to state
        Integer[] appleState = new Integer[mazeSize.x * mazeSize.y];
        Arrays.fill(appleState, 0);
        appleState[apple.y * mazeSize.x + apple.x] = 1;
        Collections.addAll(state, appleState);
        // add disappearTime to state
        Integer[] appleTimes = new Integer[mazeSize.x * mazeSize.y];
        Arrays.fill(appleTimes, 0);
        int appleTime=Math.round(existTime/1000);
        for(int i=0;i<appleTime;i++){
            appleTimes[i] = 1;
        }
        Collections.addAll(state, appleTimes);
        // add game time to state
        Integer[] gameTimes = new Integer[mazeSize.x * mazeSize.y];
        Arrays.fill(gameTimes, 0);
        int gameT = Math.round((System.currentTimeMillis() - gameTime) / 1000);
        for(int i=0;i<gameT;i++){
            gameTimes[i] = 1;
        }
        Collections.addAll(state, gameTimes);
        // add body state,include head info
        Integer[] snakeBodyState = new Integer[mazeSize.x * mazeSize.y];
        Arrays.fill(snakeBodyState, 0);
        for (int i = 0; i < bodys.length; i++) {
            int index = ((Coordinate) bodys[i]).y * mazeSize.x + ((Coordinate) bodys[i]).x;
            snakeBodyState[index] = i + 1;
        }
        Collections.addAll(state, snakeBodyState);
        // add opponent body state,include head info
        Integer[] opponentBodyState = new Integer[mazeSize.x * mazeSize.y];
        Arrays.fill(opponentBodyState, 0);
        for (int i = 0; i < opponentBodys.length; i++) {
            int index = ((Coordinate) opponentBodys[i]).y * mazeSize.x + ((Coordinate) opponentBodys[i]).x;
            opponentBodyState[index] = i + 1;
        }
        Collections.addAll(state, opponentBodyState);

        return state;
    }

    List<List<Integer>> loadDataList(String pathString){
        List<List<Integer>> states = new ArrayList<>();
        // load data from csv
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new java.io.FileReader(pathString));
            String line = null;
            while ((line = reader.readLine()) != null) {
                List<Integer> temp=new ArrayList<>();
                String[] item = line.split(",");
                for (int i = 0; i < item.length; i++) {
                    temp.add(Integer.parseInt(item[i]));
                }
                states.add(temp);
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            output("loadDataList error:" + e.getMessage());
        }

        return states;
    }

    public static void saveDataToCSV(List<Integer> state, String fileName) {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(fileName, true));

            writer.write(state.toString().substring(1, state.toString().length() - 1));
            writer.newLine();

            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            output("saveDataToCSV error:" + e.getMessage());
        }
    }

    public static void output(String text) {
        FileWriter fw;
        try {
            // if file is not exist,create it
            fw = new FileWriter(LOG_PATH, true);
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        try {
            fw.write(text + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                fw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}