package student.model;

import java.io.FileWriter;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import snakes.Bot;
import snakes.Coordinate;
import snakes.Direction;
import snakes.Snake;

public class SLModelBot_Rule implements Bot {
    private static final String LOG_PATH = "ModelRunner_Rule.log";

    private static final Direction[] DIRECTIONS = new Direction[] { Direction.UP, Direction.DOWN, Direction.LEFT,
            Direction.RIGHT };

    HttpClient httpClient = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_2)
            .connectTimeout(Duration.ofSeconds(1))
            .build();

    // record last apple location
    public Coordinate lastApple = null;

    // record apple generate time and location
    private long appleTime = System.currentTimeMillis();

    // record bot start time
    private long gameTime = System.currentTimeMillis();

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
                throw new IllegalArgumentException("Invalid direction: " + direction);
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
                throw new IllegalArgumentException("Invalid number: " + number);
        }
    }

    /**
     * Choose the direction
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
        // get valid direction
        Direction[] valDirections = getNotLosingDirections(snake, opponent, mazeSize);

        // get apple disappear time
        if (lastApple != null && !lastApple.equals(apple)) {
            appleTime = System.currentTimeMillis();
        }
        long existTime = System.currentTimeMillis() - appleTime;

        List<Integer> state = generateOneHotState(snake, opponent, mazeSize, apple, existTime);

        Direction resDirection = null;
        // get dirction
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("http://127.0.0.1:5000/predict"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString("{\"input\":[" + state + "]}"))
                    .build();

            CompletableFuture<Direction> directionFuture = httpClient
                    .sendAsync(request, HttpResponse.BodyHandlers.ofString())
                    .thenApply(HttpResponse::body)
                    .thenApply(responseBody -> {
                        JsonObject jsonObject = JsonParser.parseString(responseBody).getAsJsonObject();
                        List<Direction> directions = new ArrayList<>();
                        JsonArray jsonArray = jsonObject.get("indices").getAsJsonArray().get(0).getAsJsonArray();
                        jsonArray.forEach(jsonElement -> {
                            directions.add(numberToDirection(jsonElement.getAsInt()));
                        });

                        for (Direction direction : directions) {
                            if (Arrays.asList(valDirections).contains(direction)) {
                                return direction;
                            }
                        }
                        return null;
                    });
            resDirection = directionFuture.join();

        } catch (Exception e) {
            e.printStackTrace();
            output("Get direction error:" + e.getMessage());
        }

        // save state and reward on the file
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

    public static Direction[] getNotLosingDirections(Snake snake, Snake opponent, Coordinate mazeSize) {
        Coordinate head = snake.getHead();
        /*
         * Get the coordinate of the second element of the snake's body
         * to prevent going backwards
         */
        Coordinate afterHeadNotFinal = null;
        if (snake.body.size() >= 2) {
            Iterator<Coordinate> it = snake.body.iterator();
            it.next();
            afterHeadNotFinal = it.next();
        }
        final Coordinate afterHead = afterHeadNotFinal;

        Direction[] validMoves = Arrays.stream(DIRECTIONS)
                .filter(d -> !head.moveTo(d).equals(afterHead)) // Filter out the backwards move
                .sorted()
                .toArray(Direction[]::new);

        /* Just naïve greedy algorithm that tries not to die at each moment in time */
        return Arrays.stream(validMoves)
                .filter(d -> head.moveTo(d).inBounds(mazeSize)) // Don't leave maze
                .filter(d -> !opponent.elements.contains(head.moveTo(d))
                        || head.moveTo(d).equals(opponent.body.getLast())) // Don't collide with opponent...
                .filter(d -> !snake.elements.contains(head.moveTo(d)) || head.moveTo(d).equals(snake.body.getLast())) // and
                                                                                                                      // yourself
                .sorted()
                .toArray(Direction[]::new);
    }
}