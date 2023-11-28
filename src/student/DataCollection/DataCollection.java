package student.DataCollection;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;

import negasnake.NegaSnake;
import snakes.Bot;
import snakes.Coordinate;
import snakes.Direction;
import snakes.Snake;
import snakes.SnakesUIMain;

public class DataCollection implements Bot {
    private static final String LOG_PATH = "DataCollection.log";
    private static final String DATA_PATH = "data//DataCollection.csv";

    // record apple generate time and location
    private long appleTime = System.currentTimeMillis();

    // record bot start time
    private long gameTime = System.currentTimeMillis();

    // info
    public Coordinate lastApple = null;

    // last state
    List<Integer> lastState = new ArrayList<>();

    NegaSnake negaSnake = new NegaSnake();

    // data cache
    public static LoadingCache<String, List<List<Integer>>> dataCache = CacheBuilder.newBuilder()
            .maximumSize(1000)
            .build(new CacheLoader<String, List<List<Integer>>>() {
                @Override
                public List<List<Integer>> load(String path) throws Exception {
                    return loadDataList(path);
                }
            });

    public DataCollection() {
        // init data cache
        try {
            dataCache.get(DATA_PATH);
        } catch (ExecutionException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        appleTime = System.currentTimeMillis();
        gameTime = System.currentTimeMillis();

    }

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
        // get apple disappear time
        if (lastApple != null && !lastApple.equals(apple)) {
            appleTime = System.currentTimeMillis();
        }
        long existTime = System.currentTimeMillis() - appleTime;

        List<Integer> state = generateOneHotState(snake, opponent, mazeSize, apple, existTime);
        // get dirction from negaSnake
        try {
            resDirection = negaSnake.chooseDirection(snake, opponent, mazeSize, apple);
        } catch (Exception e) {
            e.printStackTrace();
            output("NegaSnake error:" + e.getMessage());
        }
        // save state and reward on the file
        List<Integer> datas = new ArrayList<>();
        datas.addAll(state);
        datas.add(directionToNumber(resDirection));
        Thread thread = new Thread(() -> {
            saveDataToCSV(datas, DATA_PATH);
        });
        thread.start();
        System.out.println("DataCollection time:" + (System.currentTimeMillis() - startTime) + "ms");
        lastApple = apple;
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
        int appleTime = Math.round(existTime / 1000);
        for (int i = 0; i < appleTime; i++) {
            appleTimes[i] = 1;
        }
        Collections.addAll(state, appleTimes);
        // add game time to state
        Integer[] gameTimes = new Integer[mazeSize.x * mazeSize.y];
        Arrays.fill(gameTimes, 0);
        int gameT = Math.round((System.currentTimeMillis() - gameTime) / 1000);
        for (int i = 0; i < gameT; i++) {
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

    static List<List<Integer>> loadDataList(String pathString) {
        List<List<Integer>> states = new ArrayList<>();
        // check file exist
        if (!java.nio.file.Files.exists(java.nio.file.Paths.get(pathString))) {
            return states;
        }
        // load data from csv
        try (BufferedReader reader = new BufferedReader(new java.io.FileReader(pathString))) {
            String line;
            while ((line = reader.readLine()) != null) {
                List<Integer> temp = new ArrayList<>();
                String[] items = line.split(",");
                for (String item : items) {
                    try {
                        temp.add(Integer.parseInt(item.trim()));
                    } catch (NumberFormatException e) {
                        throw new RuntimeException("loadDataList error:" + e.getMessage());
                    }
                }
                states.add(temp);
            }
        } catch (Exception e) {
            e.printStackTrace();
            output("loadDataList error:" + e.getMessage());
        }

        return states;
    }

    public static synchronized void saveDataToCSV(List<Integer> state, String fileName) {
        try {
            // Load data from cache
            if (dataCache.get(fileName).contains(state)) {
                return;
            }
            dataCache.get(fileName).add(state);
            // Use try-with-resources for better resource management
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName, true))) {
                String csvLine = state.stream()
                        .map(String::valueOf)
                        .collect(Collectors.joining(","));
                writer.write(csvLine);
                writer.newLine();
            }
        } catch (Exception e) {
            e.printStackTrace();
            output("saveDataToCSV error with file '" + fileName + "': " + e.getMessage());
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

    public static void main(String[] args) {
        String[] botNames = { "student.BasicBot", "student.DataCollection.DataCollection" };

        int threadNum = 8;

        while (true) {
            int threadsToWaitFor = (int) Math.ceil(threadNum * 0.8);
            CountDownLatch latch = new CountDownLatch(threadsToWaitFor);
            for (int i = 0; i < threadNum; i++) {
                Thread thread = new Thread(() -> {
                    try {
                        SnakesUIMain.main(botNames);
                    } catch (NoSuchMethodException | InstantiationException | IllegalAccessException
                            | InvocationTargetException
                            | InterruptedException | IOException e) {
                        e.printStackTrace();
                    } finally {
                        latch.countDown();
                    }
                });
                thread.start();
            }

            // wait almost thread finish
            try {
                latch.await();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.out.println("Finish one round,collection data size:" + dataCache.getIfPresent(DATA_PATH).size());
            // exit if data size > 500000
            if (dataCache.getIfPresent(DATA_PATH).size() > 500000) {
                break;
            }
        }
    }
}