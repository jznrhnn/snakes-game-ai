package student;

import snakes.Bot;
import snakes.Coordinate;
import snakes.Direction;
import snakes.Snake;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Implements basic snake bot
 */
public class BasicBot implements Bot {
    private static final String LOG_PATH = "BasicBotLogs.log";

    private static final Direction[] DIRECTIONS = new Direction[] { Direction.UP, Direction.DOWN, Direction.LEFT,
            Direction.RIGHT };

    /**
     * Outputs text to stdout and file
     *
     * @param text text that should be displayed
     */
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
            if (this == o)
                return true;
            if (o == null || getClass() != o.getClass())
                return false;

            Node node = (Node) o;

            if (x != node.x)
                return false;
            if (y != node.y)
                return false;
            if (g != node.g)
                return false;
            if (h != node.h)
                return false;
            return true;
            // return parent != null ? parent.equals(node.parent) : node.parent == null;
        }

        @Override
        public int hashCode() {
            int result = x;
            result = 31 * result + y;
            result = 31 * result + g;
            result = 31 * result + h;
            // result = 31 * result + (parent != null ? parent.hashCode() : 0);
            return result;
        }

        /**
         * for full path (include detour)
         * 
         * @param o
         * @return
         */
        public boolean fullEquals(Object o) {
            if (this == o)
                return true;
            if (o == null || getClass() != o.getClass())
                return false;

            Node node = (Node) o;

            if (x != node.x)
                return false;
            if (y != node.y)
                return false;
            if (g != node.g)
                return false;
            if (h != node.h)
                return false;
            return parent != null ? parent.equals(node.parent) : node.parent == null;
        }

        public int fullHashCode() {
            int result = x;
            result = 31 * result + y;
            result = 31 * result + g;
            result = 31 * result + h;
            result = 31 * result + (parent != null ? parent.hashCode() : 0);
            return result;
        }

        @Override
        protected Object clone() throws CloneNotSupportedException {
            return super.clone();
        }
    }

    public static int heuristic(Coordinate snake, Coordinate apple) {
        return Math.abs(snake.x - apple.x) + Math.abs(snake.y - apple.y);
    }

    public void printList(List<int[]> list) {
        StringBuilder result = new StringBuilder();
        for (int[] ints : list) {
            result.append(Arrays.toString(ints));
        }
        output(result.toString());
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
        try {
            Direction result = null;
            // log current snake and apple position, opponent snake position
            output("------------------" + new Date() + "------------------");
            output("Snake: " + snake.getHead());
            output("snake body" + snake.body);
            output("Opponent: " + opponent.getHead());
            output("Opponent body" + opponent.body);
            output("Apple: " + apple);

            long startTime = System.currentTimeMillis();
            List<int[]> path = getPath(snake, opponent, mazeSize, apple);
            if (path.isEmpty()) {
                output("Can't find path to apple,Go to the tail keep safe");
                path = getPath(snake, opponent, mazeSize, snake.body.getLast());
                if (path.isEmpty()) {
                    output("Can't find path to tail");
                    Direction[] notLosingDirections = getNotLosingDirections(snake, opponent, mazeSize);
                    Direction[] noCollDirections = voidCollision(snake, opponent, mazeSize,
                            notLosingDirections);
                    if (noCollDirections.length > 0) {
                        result = noCollDirections[0];
                    } else {
                        output("Can't find safe direction");
                        result = snake.getHead().getDirection(snake.body.getLast());
                    }
                } else {
                    result = snake.getHead().getDirection(new Coordinate(path.get(1)[0], path.get(1)[1]));
                }
            } else {
                result = snake.getHead().getDirection(new Coordinate(path.get(1)[0], path.get(1)[1]));
            }
            output(result.toString());
            output("Time cost: " + (System.currentTimeMillis() - startTime) + "ms");
            return result;
        } catch (Exception e) {
            e.printStackTrace();
            output("Error");
            throw e;
            // return Direction.UP;
        }

    }

    /**
     * get path from snake to apple
     * 
     * @param snake
     * @param opponent
     * @param mazeSize
     * @param apple
     * @return
     */
    public List<int[]> getPath(Snake snake, Snake opponent, Coordinate mazeSize, Coordinate apple) {
        // start time of decision
        final long startTime = System.currentTimeMillis();

        // A* algorithm
        PriorityQueue<Node> openSet = new PriorityQueue<>(Comparator.comparingInt(Node::f));
        Map<Node, Integer> gValues = new HashMap<>();

        Node start = new Node(snake.getHead().x, snake.getHead().y, 0,
                Math.abs(snake.getHead().x - apple.x) + Math.abs(snake.getHead().y - apple.y), null);
        openSet.add(start);
        gValues.put(start, 0);
        // for path log
        List<int[]> path = new ArrayList<>();
        while (!openSet.isEmpty()) {
            Node current = openSet.poll();
            // over time limit
            if (System.currentTimeMillis() - startTime > 920) {
                output("Time out");
                output("Time: " + (System.currentTimeMillis() - startTime) + "ms");
                output(current.x + " " + current.y + " " + current.f());
                path = new ArrayList<>();
                break;
            }

            // get current Node all valid moves
            // update snake to current position
            Snake newSnake = snake.clone();
            Node newNode = current;
            List<Node> moves = new ArrayList<>();
            while (newNode.parent != null) {
                moves.add(newNode);
                newNode = newNode.parent;
            }
            // reverse moves
            Collections.reverse(moves);
            for (Node move : moves) {
                Direction dirction = newSnake.getHead().getDirection(new Coordinate(move.x, move.y));
                if (dirction == null) {
                    continue;
                }
                newSnake.moveTo(dirction, false);
            }
            Direction[] notLosing = getNotLosingDirections(newSnake, opponent, mazeSize);
            Direction[] voidCollision = voidCollision(newSnake, opponent, mazeSize, notLosing);

            if (current.x == apple.x && current.y == apple.y) {
                // snake will be safety when apples be eaten
                // get notlosing directions
                // if (notLosing.length <= 0) {
                // output("Can't find safe path to apple");
                // continue;
                // }
                // Build path
                while (current != null) {
                    path.add(new int[] { current.x, current.y });
                    current = current.parent;
                }

                Collections.reverse(path);
                // log time,path
                printList(path);

                // Keep path to apple is safe
                if (!snake.body.getLast().equals(apple)) {
                    newSnake = snake.clone();
                    // update snake by path
                    for (int[] p : path) {
                        Direction dirction = newSnake.getHead().getDirection(new Coordinate(p[0], p[1]));
                        if (dirction == null) {
                            continue;
                        }
                        newSnake.moveTo(dirction, false);
                    }

                    // make sure snake can go to tail
                    List<int[]> safePath = getPath(newSnake, opponent, mazeSize, newSnake.body.getLast());
                    if (safePath.isEmpty()) {
                        output("Can't find safe path to tail");
                        output("newSnake: " + newSnake.body);
                        output("Opponent: " + opponent.body);
                        output("newApple: " + newSnake.body.getLast());
                        return new ArrayList<>();
                    }
                }
                break;
            }

            // add all valid moves to openSet
            for (Direction neighbor : voidCollision) {
                Coordinate head = new Coordinate(current.x, current.y);
                int newX = head.moveTo(neighbor).x;
                int newY = head.moveTo(neighbor).y;
                int newG = current.g + 1;

                Node neighborNode = new Node(newX, newY, newG, heuristic(new Coordinate(newX, newY), apple),
                        current);

                if (!gValues.containsKey(neighborNode) || newG < gValues.get(neighborNode)) {
                    openSet.add(neighborNode);
                    gValues.put(neighborNode, newG);
                }
                // if (!gValues.containsKey(neighborNode.fullHashCode()) || newG <
                // gValues.get(neighborNode.fullHashCode())) {
                // openSet.add(neighborNode);
                // gValues.put(neighborNode.fullHashCode(), newG);
                // }
            }
        }
        // log time
        output("Get path time: " + (System.currentTimeMillis() - startTime) + "ms");
        return path;
    }

    /**
     * avoid collision while opponent is longer than snake
     * Get the directions that will not make the snake collide with the wall or its
     * own body
     */
    public static Direction[] voidCollision(Snake snake, Snake opponent, Coordinate mazeSize, Direction[] notLosing) {
        Coordinate head = snake.getHead();
        // avoid collision while opponent is longer than snake
        Direction[] voidCollision = null;
        if (opponent.body.size() > snake.body.size()) {
            Direction[] opponentDirections = getNotLosingDirections(opponent, snake, mazeSize);
            List<Coordinate> opponentHeads = new ArrayList<>();
            for (Direction opponentDirection : opponentDirections) {
                Coordinate opponentHead = opponent.getHead().moveTo(opponentDirection);
                opponentHeads.add(opponentHead);
            }

            voidCollision = Arrays.stream(notLosing)
                    .filter(direction -> !opponentHeads.contains(head.moveTo(direction)))
                    .toArray(Direction[]::new);
        }
        // if voidCollision is null, return notLosing
        if (voidCollision == null || voidCollision.length == 0) {
            return notLosing;
        }
        return voidCollision;
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
