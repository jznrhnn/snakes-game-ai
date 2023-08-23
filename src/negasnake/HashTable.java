/* Copyright (c) 2021, Gijs Pennings
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package negasnake;

import java.util.Arrays;

import static negasnake.Constants.INF;
import static negasnake.Constants.TABLE_SIZE;

/**
 * Fixed size hash table for move ordering using open addressing and linear probing.
 */
public final class HashTable {

    private final long[] array = new long[TABLE_SIZE];

    public void clear() {
        Arrays.fill(array, 0);
    }

    /**
     * Returns value that is mapped to by {@code key > 0} (or {@code INF}).
     */
    public int get(final int key) {
        for (int i = key; true; i++) {
            i %= TABLE_SIZE;
            final long x = array[i];
            if (x == 0) return INF;
            if (key == (int) x) return (int) (x >>> 32);
        }
    }

    /**
     * Associates {@code val} with {@code key > 0}. Any existing pairing is overwritten.
     */
    public void set(final int key, final int val) {
        final long x = key | (long) val << 32;
        for (int i = key; true; i++) {
            i %= TABLE_SIZE;
            final int k = (int) array[i];
            if (k == 0 || k == key) { array[i] = x; return; }
        }
    }

}
