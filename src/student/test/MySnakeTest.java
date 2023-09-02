package student.test;

import org.junit.jupiter.api.Test;
import snakes.Coordinate;
import snakes.Direction;
import snakes.Snake;
import student.BasicBot;

import java.util.Arrays;

/**
 * @author f
 * @date 8/29/2023 9:35 PM
 */
public class MySnakeTest {
    //    Test output function
    @Test
    public void outputTest() {
        BasicBot.output("test1");
        BasicBot.output("1");
        BasicBot.output("2");
        BasicBot.output("3");
    }

    @Test
    public void basicSnakeTest(){
//        init snake,apple,mazeSize
        Coordinate mazeSize = new Coordinate(14, 14);
        Coordinate head0 = new Coordinate(10, 2);
        Direction tailDirection0 = Direction.DOWN;


        Snake snake = new Snake(head0, tailDirection0, 4, mazeSize);

        Coordinate head1 = new Coordinate(4, 0);
        Direction tailDirection1= Direction.LEFT;


        Snake opponent = new Snake(head1, tailDirection1, 5, mazeSize);

        BasicBot basicBot = new BasicBot();
        basicBot.chooseDirection(snake, opponent, mazeSize, new Coordinate(3, 0));
    }

    @Test
    public void test(){
        System.out.println(Arrays.toString(test1()));
    }

    Direction[] test1(){
        Direction[] validMoves = new Direction[]{Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT};
        return Arrays.stream(validMoves)
                .filter(d -> d==Direction.UP)             // Don't leave maze
                .sorted()
                .toArray(Direction[]::new);
    }
}
