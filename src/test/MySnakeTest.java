package test;

import org.junit.jupiter.api.Test;
import snakes.Coordinate;
import snakes.Direction;
import snakes.Snake;
import snakes.SnakeGame;
import snakes.SnakesWindow;
import student.BasicBot;

import javax.xml.crypto.Data;
import java.util.*;

import static student.BasicBot.getNotLosingDirections;
import static student.BasicBot.voidCollision;

/**
 * @author f
 * @date 8/29/2023 9:35 PM
 */
public class MySnakeTest {

    public static void main(String[] args) {
        Coordinate mazeSize = new Coordinate(14, 14);
        Deque<Coordinate> body = new java.util.LinkedList<>();
        HashSet<Coordinate> elements = new HashSet<>();

        //String snakeString = "(11, 3), (11, 2), (11, 1), (11, 0), (12, 0), (12, 1), (13, 1), (13, 2), (13, 3), (13, 4), (13, 5), (12, 5), (11, 5), (11, 6), (11, 7), (11, 8), (10, 8), (10, 7), (10, 6), (10, 5), (10, 4), (10, 3), (10, 2)";
        String snakeString = "(0, 12), (1, 12), (2, 12), (3, 12), (4, 12), (4, 11), (4, 10), (4, 9), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8), (9, 7), (10, 7), (11, 7), (12, 7), (12, 6), (11, 6), (10, 6), (9, 6), (9, 5), (9, 4), (9, 3), (9, 2), (8, 2), (7, 2), (6, 2), (6, 1), (5, 1), (5, 2), (4, 2), (4, 3), (3, 3), (3, 4), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10)";
        stringToList(body, snakeString);

        // copy element to body
        elements.addAll(body);

        // Direction tailDirection0 = Direction.DOWN;
        Snake snake = new Snake(mazeSize, elements, body);

        // oppnent
        Deque<Coordinate> body1 = new java.util.LinkedList<>();
        HashSet<Coordinate> elements1 = new HashSet<>();

        String string = "(10, 11), (9, 11), (9, 10), (10, 10)";
        stringToList(body1, string);
        // copy element1 to body1
        elements1.addAll(body1);

        Snake opponent = new Snake(mazeSize, elements1, body1);

        // apple
        Coordinate apple = new Coordinate(2, 13);

        BasicBot basicBot = new BasicBot();

        SnakeGame game = new SnakeGame(mazeSize, snake, opponent, basicBot, basicBot);
        game.appleCoordinate = apple;
        SnakesWindow window = new SnakesWindow(game);

        List<int[]> path = basicBot.getPath(snake, opponent, mazeSize, snake.body.getLast());
        StringBuilder result = new StringBuilder();
        for (int[] ints : path) {
            result.append(Arrays.toString(ints));
        }
        System.out.println(result);
        Direction direction = basicBot.chooseDirection(snake, opponent, mazeSize, apple);
        System.out.println(direction);
    }

    private static void stringToList(Deque<Coordinate> body, String snakeString) {
        String[] snakeStringArray = snakeString.replace("(", "").replace(")", "").split(", ");
        for (int i = 0; i < snakeStringArray.length; i = i + 2) {
            Coordinate coordinate = new Coordinate(Integer.parseInt(snakeStringArray[i]),
                    Integer.parseInt(snakeStringArray[i + 1]));
            body.add(coordinate);
        }
    }

    @Test
    public void testGetPath() {
        Coordinate mazeSize = new Coordinate(14, 14);
        Deque<Coordinate> body = new java.util.LinkedList<>();
        HashSet<Coordinate> elements = new HashSet<>();

        String snakeString = "(4, 2), (3, 2), (2, 2), (2, 1)";

        stringToList(body, snakeString);

        // copy element to body
        elements.addAll(body);

        // Direction tailDirection0 = Direction.DOWN;
        Snake snake = new Snake(mazeSize, elements, body);

        // oppnent
        Deque<Coordinate> body1 = new java.util.LinkedList<>();
        HashSet<Coordinate> elements1 = new HashSet<>();

        String string = "(10, 12), (10, 11), (11, 11)";
        stringToList(body1, string);
        // copy element1 to body1
        elements1.addAll(body1);

        Snake opponent = new Snake(mazeSize, elements1, body1);

        // apple
        Coordinate apple = new Coordinate(3, 4);

        BasicBot basicBot = new BasicBot();

        List<int[]> path = basicBot.getPath(snake, opponent, mazeSize, apple);
        Direction direction = basicBot.chooseDirection(snake, opponent, mazeSize, apple);

        StringBuilder result = new StringBuilder();
        for (int[] ints : path) {
            result.append(Arrays.toString(ints));
        }
        assert "[4, 2][4, 3][4, 4][3, 4]".equals(result.toString());
        assert direction == Direction.UP;
    }

    @Test
    public void testGetPathGraphics() {
        Coordinate mazeSize = new Coordinate(14, 14);
        Deque<Coordinate> body = new java.util.LinkedList<>();
        HashSet<Coordinate> elements = new HashSet<>();

        String snakeString = "(9, 9), (9, 8), (9, 7), (9, 6), (9, 5), (9, 4), (8, 4), (8, 5), (8, 6), (7, 6), (7, 7), (7, 8), (7, 9), (6, 9), (6, 10), (5, 10), (4, 10), (4, 9), (4, 8), (4, 7), (4, 6), (5, 6), (5, 5), (6, 5), (6, 4), (6, 3)";
        stringToList(body, snakeString);

        // copy element to body
        elements.addAll(body);

        // Direction tailDirection0 = Direction.DOWN;
        Snake snake = new Snake(mazeSize, elements, body);

        // oppnent
        Deque<Coordinate> body1 = new java.util.LinkedList<>();
        HashSet<Coordinate> elements1 = new HashSet<>();

        String string = "(10, 12), (10, 11), (11, 11)";
        stringToList(body1, string);
        // copy element1 to body1
        elements1.addAll(body1);

        Snake opponent = new Snake(mazeSize, elements1, body1);

        // apple
        Coordinate apple = new Coordinate(6, 3);

        BasicBot basicBot = new BasicBot();

        SnakeGame game = new SnakeGame(mazeSize, snake, opponent, basicBot, basicBot);
        game.appleCoordinate = apple;
        SnakesWindow window = new SnakesWindow(game);
        List<int[]> path = basicBot.getPath(snake, opponent, mazeSize, apple);
        Direction direction = basicBot.chooseDirection(snake, opponent, mazeSize, apple);

        StringBuilder result = new StringBuilder();
        for (int[] ints : path) {
            result.append(Arrays.toString(ints));
        }
        System.out.println(result);

    }

    @Test
    public void testVoidCollision() {
        Coordinate mazeSize = new Coordinate(14, 14);
        Deque<Coordinate> body = new java.util.LinkedList<>();
        HashSet<Coordinate> elements = new HashSet<>();

        String snakeString = "(9, 13), (8, 13), (8, 12), (8, 11), (8, 10), (9, 10), (9, 9), (9, 8), (9, 7), (9, 6), (9, 5), (9, 4), (8, 4), (8, 5), (8, 6), (7, 6), (7, 7), (7, 8), (7, 9), (6, 9), (6, 10), (5, 10), (4, 10), (4, 9), (4, 8), (4, 7)";
        stringToList(body, snakeString);

        // copy element to body
        elements.addAll(body);

        // Direction tailDirection0 = Direction.DOWN;
        Snake snake = new Snake(mazeSize, elements, body);

        // oppnent
        Deque<Coordinate> body1 = new java.util.LinkedList<>();
        HashSet<Coordinate> elements1 = new HashSet<>();

        String string = "(10, 12), (10, 11), (11, 11)";
        stringToList(body1, string);
        // copy element1 to body1
        elements1.addAll(body1);

        Snake opponent = new Snake(mazeSize, elements1, body1);

        Direction[] notLosingDirections = getNotLosingDirections( snake,opponent, mazeSize);

        Direction[] noCollDirections = voidCollision(snake, opponent, mazeSize, notLosingDirections);
        System.out.println(Arrays.toString(notLosingDirections));
        System.out.println(Arrays.toString(noCollDirections));

    }

    @Test
    void test1() {
        int[] ints = new int[4];
        System.out.println(Arrays.toString(ints));
    }
}
