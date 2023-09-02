package student;

import snakes.Bot;
import snakes.Coordinate;
import snakes.Direction;
import snakes.Snake;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Implements basic snake bot
 */
public class BasicBot implements Bot {
    private static final String LOG_PATH = "BasicBotLogs";

    private static final Direction[] DIRECTIONS = new Direction[] {Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT};

    /**
     * Outputs text to stdout and file
     *
     * @param text text that should be displayed
     */
    public static void output(String text) {
        FileWriter fw;
        try {
            //if file is not exist,create it
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

    /**
     * delete log file
     */
    public void clearLog() {
        try {
            File logFile = new File(LOG_PATH);
            if (logFile.exists()) {
                if (logFile.delete()) {
                    System.out.println("Deleted log file: " + LOG_PATH);
                } else {
                    System.err.println("Failed to delete log file: " + LOG_PATH);
                }
            } else {
                System.out.println("Log file not found: " + LOG_PATH);
            }
        } catch (SecurityException e) {
            System.err.println("Permission denied when deleting log files.");
        } catch (Exception e) {
            System.err.println("An error occurred while deleting log files: " + e.getMessage());
        }
    }

    static class Node {
        int x, y, g, h;
        Node parent;

        public Node(int x, int y, int g, int h, Node parent) {
            this.x = x;
            this.y = y;
            this.g = g;
            this.h = h;
            this.parent = parent;
        }

        public int f() {
            return g + h;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            Node node = (Node) o;

            if (x != node.x) return false;
            if (y != node.y) return false;
            if (g != node.g) return false;
            if (h != node.h) return false;
            return parent != null ? parent.equals(node.parent) : node.parent == null;
        }

        @Override
        public int hashCode() {
            int result = x;
            result = 31 * result + y;
            result = 31 * result + g;
            result = 31 * result + h;
            result = 31 * result + (parent != null ? parent.hashCode() : 0);
            return result;
        }
    }

    public static int heuristic(Coordinate snake,Coordinate apple) {
        return Math.abs(snake.x - apple.x) + Math.abs(snake.y - apple.y);
    }

    public void printList(List<int[]> list){
        StringBuilder result = new StringBuilder();
        for (int[] ints : list) {
            result.append(Arrays.toString(ints));
        }
        output(result.toString());
    }

    /**
     * Choose the direction (not rational - silly)
     * @param snake    Your snake's body with coordinates for each segment
     * @param opponent Opponent snake's body with coordinates for each segment
     * @param mazeSize Size of the board
     * @param apple    Coordinate of an apple
     * @return Direction of bot's move
     */
    @Override
    /* choose the direction (stupidly) */
    public Direction chooseDirection(Snake snake, Snake opponent, Coordinate mazeSize, Coordinate apple) {
        //start time
        final long startTime = System.currentTimeMillis();
        //log current snake and apple position, opponent snake position
        output("------------------" + startTime + "------------------");
        output("Snake: " + snake.getHead());
        output("snake body"+snake.body);
        output("Opponent: " + opponent.getHead());
        output("Opponent body"+opponent.body);
        output("Apple: " + apple);
        Direction result = null;
        /* Get the coordinate of the second element of the snake's body
         * to prevent going backwards */
        Coordinate afterHeadNotFinal = null;
        if (snake.body.size() >= 2) {
            Iterator<Coordinate> it = snake.body.iterator();
            it.next();
            afterHeadNotFinal = it.next();
        }
        final Coordinate afterHead = afterHeadNotFinal;

        //A* algorithm
        PriorityQueue<Node> openSet = new PriorityQueue<>(Comparator.comparingInt(Node::f));
        Map<Node, Integer> gValues = new HashMap<>();

        Node start = new Node(snake.getHead().x, snake.getHead().y, 0, Math.abs(snake.getHead().x - apple.x) + Math.abs(snake.getHead().y - apple.y), null);
        openSet.add(start);
        gValues.put(start, 0);
       //for path log
        boolean flag = false;
        while (!openSet.isEmpty()) {
            Node current = openSet.poll();
            //over time limit
            if (System.currentTimeMillis() - startTime > 930) {
                output("Time out");
                output("Time: " + (System.currentTimeMillis() - startTime) + "ms");
                output(current.x + " " + current.y + " " + current.f());
                break;
            }

            //get current Node all valid moves
            Coordinate head = new Coordinate(current.x, current.y);
            Direction[] notLosing = getNotLosingDirections(snake, opponent, mazeSize, afterHead,head);
            Direction[] voidCollision = voidCollision(snake, opponent, mazeSize, afterHead, head, notLosing);

            if (current.x == apple.x && current.y == apple.y) {
                //snake will be safety when apples be eaten
                //get notlosing directions
                if (notLosing.length<=0) {
                    output("Can't find path to apple");
                    continue;
                }
                List<int[]> path = new ArrayList<>();
                while (current != null) {
                    path.add(new int[]{current.x, current.y});
                    current = current.parent;
                }
                Collections.reverse(path);
                //log time,path
                printList(path);
                result = snake.getHead().getDirection(new Coordinate(path.get(1)[0], path.get(1)[1]));
                flag = true;
                break;
            }

            //add all valid moves to openSet
            for (Direction neighbor : voidCollision) {
                int newX = head.moveTo(neighbor).x;
                int newY = head.moveTo(neighbor).y;
                if (newX < 0 || newY < 0 || newX >= mazeSize.x || newY >= mazeSize.y) {
                    System.out.println("Out of bound");
                }
                int newG = current.g + 1;

                Node neighborNode = new Node(newX, newY, newG, heuristic(new Coordinate(newX, newY), apple), current);

                if (!gValues.containsKey(neighborNode) || newG < gValues.get(neighborNode)) {
                    openSet.add(neighborNode);
                    gValues.put(neighborNode, newG);
                }
            }
        }
        //log time
        output("Time: " + (System.currentTimeMillis() - startTime) + "ms");
        if (!flag){
            output("Can't find path to apple");
            Direction[] notLosing = getNotLosingDirections(snake, opponent, mazeSize, afterHead, snake.getHead());
            result = notLosing[0];
        }
        output(result.toString());
        return result;
    }

    /**
     * avoid collision while opponent is longer than snake
     * Get the directions that will not make the snake collide with the wall or its own body
     */
    private Direction[] voidCollision(Snake snake, Snake opponent, Coordinate mazeSize, Coordinate afterHead, Coordinate head, Direction[] notLosing) {
        if (opponent.body.size() > snake.body.size()) {
            Direction[] opponentDirections=getNotLosingDirections(opponent, snake, mazeSize, afterHead, opponent.getHead());
            List<Coordinate> opponentHeads=new ArrayList<>();
            for (Direction opponentDirection : opponentDirections) {
                Coordinate opponentHead = opponent.getHead().moveTo(opponentDirection);
                opponentHeads.add(opponentHead);
            }
            for (Direction direction : notLosing) {
                Coordinate newHead = head.moveTo(direction);
                if (opponentHeads.contains(newHead)) {
                    notLosing = Arrays.stream(notLosing).filter(d -> d != direction).toArray(Direction[]::new);
                }
            }
        }
        return notLosing;
    }

    private Direction[] getNotLosingDirections(Snake snake, Snake opponent, Coordinate mazeSize, Coordinate afterHead,Coordinate head) {
        Direction[] validMoves = Arrays.stream(DIRECTIONS)
                .filter(d -> !head.moveTo(d).equals(afterHead)) // Filter out the backwards move
                .sorted()
                .toArray(Direction[]::new);

        /* Just naïve greedy algorithm that tries not to die at each moment in time */
        return Arrays.stream(validMoves)
                .filter(d -> head.moveTo(d).inBounds(mazeSize))             // Don't leave maze
                .filter(d -> !opponent.elements.contains(head.moveTo(d)))   // Don't collide with opponent...
                .filter(d -> !snake.elements.contains(head.moveTo(d)))      // and yourself
                .sorted()
                .toArray(Direction[]::new);
    }


}
