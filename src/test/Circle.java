package test;

import snakes.Bot;
import snakes.Coordinate;
import snakes.Direction;
import snakes.Snake;

import java.util.Iterator;

/**
 * Sample implementation of snake bot
 */
public class Circle implements Bot {
    /**
     * Choose the direction (not rational - silly)
     * @param snake    Your snake's body with coordinates for each segment
     * @param opponent Opponent snake's body with coordinates for each segment
     * @param mazeSize Size of the board
     * @param apple    Coordinate of an apple
     * @return Direction of bot's move
     */
    @Override
    public Direction chooseDirection(Snake snake, Snake opponent, Coordinate mazeSize, Coordinate apple) {
        Coordinate head = snake.getHead();

        /* Get the coordinate of the second element of the snake's body
         * to prevent going backwards */
        Coordinate afterHeadNotFinal = null;
        if (snake.body.size() >= 2) {
            Iterator<Coordinate> it = snake.body.iterator();
            it.next();
            afterHeadNotFinal = it.next();
        }

        final Coordinate afterHead = afterHeadNotFinal;

        Direction resDirection = null;
        Direction direction=afterHead.getDirection(head);
        if(direction==Direction.LEFT){
            resDirection=Direction.UP;
        }else if(direction==Direction.UP){
            resDirection=Direction.RIGHT;
        }else if(direction==Direction.RIGHT){
            resDirection=Direction.DOWN;
        }else if(direction==Direction.DOWN){
            resDirection=Direction.LEFT;
        }
        return resDirection;
    }
}