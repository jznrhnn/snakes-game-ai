## DataCollection

The Data Collection module is designed to record a comprehensive set of features that may influence directional choices in the game environment. These features are primarily one-hot encoded and carry distinct physical meanings. Below is a detailed breakdown of the data collected:

- **Snake, Opponent Body, and Apple Information**: A set of integers representing a 14x14 grid, each encoding the presence of the snake, the opponent's body, and apples in the game environment.
- **Apple Existence Time**: Records the duration for which the current apple has been present in the game, structured as a 14x14 grid.
- **Game Start Time**:  Tracks the elapsed game time, structured as a 14x14 grid representation.
- **Snake and Opponent Body Queue Information**: A series of integers detailing the snake and opponent's body positions over time, structured as a 14x14 grid representation.

Each integer in the list is critical for the game's decision-making processes, providing valuable insights into the dynamic game environment.(Other features that may affect the choice of direction can be added as needed)

### usage

1. **Configure Settings(optional)**:
   - Modify the data source, the default sources are `BasicBot` and `NegaSnake`
   - Adjust file paths in `DataCollection.java` according to your project structure.
   - The default setting is 10 threads. Adjust the thread count in `DataCollection.java` based on your system's resources.
   - By default, the module is set to collect 500,000 data points. This can be adjusted as your requirements.

2. **Run the Application**:
   - Execute the module by running `DataCollection.main()`. 
