## DataConllection

The Data Collection module is designed to record a comprehensive set of features that may influence directional choices in the game environment. These features are primarily one-hot encoded and carry distinct physical meanings. Below is a detailed breakdown of the data collected:

- **Snake, Opponent Body, and Apple Information**: A set of integers representing a 14x14 grid, each encoding the presence of the snake, the opponent's body, and apples in the game environment.
- **Apple Existence Time**: An integer value indicating the duration for which the apple has been present in the game.
- **Game Start Time**: An integer representing the time at which the game started.
- **Snake and Opponent Body Queue Information**: A series of integers detailing the snake and opponent's body positions over time, structured as a 14x14 grid representation.

Each integer in the list is critical for the game's decision-making processes, providing valuable insights into the dynamic game environment.(Other features that may affect the choice of direction can be added as needed)

### usage

1. **Export Project as JAR File**:

2. **Configure Settings(optional)**:

   - Modify the file paths in `DataCollection.java` as needed.
   - Each thread requires approximately 235MB of memory. The default setting is 10 threads. Adjust the thread count in `DataCollection.py` based on your system's resources.

3. **Run the Application**:
