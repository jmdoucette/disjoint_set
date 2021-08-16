//! This package implements the [`DisjointSet`] data structure using a disjoint set forest.
use std::cmp::min;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::iter::FromIterator;

/// A disjoint set implemented using a disjoint set forest.
///
/// # Examples
///
/// ```
/// use disjoint_set_forest::DisjointSet;
///
/// // Type inference lets us omit an explicit type signature (which
/// // would be `DisjointSet<i32>` in this example).
/// let mut ds = DisjointSet::new();
///
/// // Adding some elements
/// ds.insert(1);
/// ds.insert(2);
/// ds.insert(3);
///
/// // All elements are currently in separate sets in the partition
/// assert!(!ds.same_set(&1, &2).unwrap());
/// assert!(!ds.same_set(&1, &3).unwrap());
/// assert!(!ds.same_set(&2, &3).unwrap());
///
/// // combining several sets
/// ds.union(&1, &2);
/// ds.union(&2, &3);
///
/// // All elements are now in the same set in the partition
/// assert!(ds.same_set(&1, &2).unwrap());
/// assert!(ds.same_set(&1, &3).unwrap());
/// assert!(ds.same_set(&2, &3).unwrap());
///
/// // iterating through the elements in arbitrary order
/// for x in ds.iter() {
///     println!("{}", x);
/// }
/// ```
#[derive(Clone, Default, Eq)]
pub struct DisjointSet<T: Hash + Eq> {
    val_to_index: HashMap<T, usize>,
    parents: Vec<usize>,
    sizes: Vec<usize>,
}

impl<T: Hash + Eq> DisjointSet<T> {
    /// Creates a new, empty `DisjointSet`.
    ///
    /// # Examples
    /// ```
    /// use disjoint_set_forest::DisjointSet;
    /// let mut ds = DisjointSet::new();
    /// ds.insert(5);
    /// ```
    pub fn new() -> Self {
        Self {
            val_to_index: HashMap::new(),
            parents: Vec::new(),
            sizes: Vec::new(),
        }
    }

    /// Creates a new, empty `DisjointSet` with the specified capacity.
    /// This preallocates enough memory for `capacity` elements,
    /// so that the `DisjointSet` does not have to be reallocated
    /// until it contains at least that many values.
    ///
    /// # Examples
    /// ```
    /// use disjoint_set_forest::DisjointSet;
    /// let mut ds = DisjointSet::with_capacity(10);
    /// ds.insert(5);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            val_to_index: HashMap::with_capacity(capacity),
            parents: Vec::with_capacity(capacity),
            sizes: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of elements in a disjoint set data structure.
    ///
    /// # Examples
    /// ```
    /// use disjoint_set_forest::DisjointSet;
    /// let mut ds = DisjointSet::new();
    /// ds.insert(5);
    /// ds.insert(10);
    ///
    /// assert_eq!(ds.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.val_to_index.len()
    }

    /// Returns `true` if the disjoint set data structure contains no elements.
    ///
    /// # Examples
    /// ```
    /// use disjoint_set_forest::DisjointSet;
    /// let mut ds = DisjointSet::new();
    ///
    /// assert!(ds.is_empty());
    ///
    /// ds.insert(5);
    /// ds.insert(10);
    ///
    /// assert!(!ds.is_empty());
    pub fn is_empty(&self) -> bool {
        self.val_to_index.is_empty()
    }

    /// Returns `true` if the disjoint set data structure contains the specified element.
    ///
    /// # Examples
    /// ```
    /// use disjoint_set_forest::DisjointSet;
    /// let mut ds = DisjointSet::new();
    /// ds.insert(5);
    /// ds.insert(10);
    ///
    /// assert!(ds.contains(&5));
    /// assert!(!ds.contains(&6));
    /// ```
    pub fn contains(&self, x: &T) -> bool {
        self.val_to_index.contains_key(x)
    }

    /// Adds a element to the disjoint set data structure as a singleton set.
    ///
    /// If the disjoint set data structure did not contain this element, `true` is returned.
    /// If the disjoint set data structure did contain this element, `false` is returned.
    ///
    /// # Examples
    /// ```
    /// use disjoint_set_forest::DisjointSet;
    /// let mut ds = DisjointSet::new();
    ///
    /// assert_eq!(ds.insert(5), true);
    /// assert_eq!(ds.insert(5), false);
    /// assert_eq!(ds.len(), 1);
    /// ```
    pub fn insert(&mut self, x: T) -> bool {
        if self.contains(&x) {
            return false;
        }

        self.parents.push(self.len());
        self.sizes.push(1);
        self.val_to_index.insert(x, self.len());
        true
    }

    /// Combines the sets containing the two specified elements.
    ///
    /// If the disjoint set data structure does not contain both elements,
    /// an error is returned and no change occurs.
    ///
    /// # Examples
    /// ```
    /// use disjoint_set_forest::DisjointSet;
    /// let mut ds = DisjointSet::new();
    /// ds.insert(1);
    /// ds.insert(2);
    /// ds.insert(3);
    ///
    /// assert!(!ds.same_set(&1, &2).unwrap());
    /// assert!(!ds.same_set(&1, &3).unwrap());
    /// assert!(!ds.same_set(&2, &3).unwrap());
    ///
    /// ds.union(&1, &2).unwrap();
    /// ds.union(&2, &3).unwrap();
    ///
    /// assert!(ds.same_set(&1, &2).unwrap());
    /// assert!(ds.same_set(&1, &3).unwrap());
    /// assert!(ds.same_set(&2, &3).unwrap());
    ///
    /// // calling union with an element not contained within the partition
    /// assert!(ds.union(&1, &4).is_err());
    /// ```
    pub fn union(&mut self, x: &T, y: &T) -> Result<(), DisjointSetError> {
        let x = *self
            .val_to_index
            .get(x)
            .ok_or(DisjointSetError::MissingElement)?;
        let y = *self
            .val_to_index
            .get(y)
            .ok_or(DisjointSetError::MissingElement)?;
        let sx = self.find_compress(x);
        let sy = self.find_compress(y);

        // x and y are already in the same set, no work needed
        if sx != sy {
            // x is in the larger set, x becomes parent of y
            if self.sizes[sx] >= self.sizes[sy] {
                self.parents[sy] = sx;
                self.sizes[sx] += self.sizes[sy];
            }
            // y is in the larger set, y becomes parent of x
            else {
                self.parents[sx] = sy;
                self.sizes[sy] += self.sizes[sx];
            }
        }
        Ok(())
    }

    /// Returns the index of the set containing the element with the given index.
    fn find(&self, x: usize) -> usize {
        let mut curr = x;
        while curr != self.parents[curr] {
            curr = self.parents[curr];
        }
        curr
    }

    /// Returns the index of the set containing the element with the given index.
    /// Compresses path using path splitting
    fn find_compress(&mut self, x: usize) -> usize {
        let mut curr = x;
        while curr != self.parents[curr] {
            self.parents[curr] = self.parents[self.parents[curr]];
            curr = self.parents[curr];
        }
        curr
    }

    /// Returns `true` if the two specified elements are contained in the same set.
    ///
    /// If the disjoint set data structure does not contain both elements, an error is returned.
    ///
    /// # Examples
    /// ```
    /// use disjoint_set_forest::DisjointSet;
    /// let mut ds = DisjointSet::new();
    /// ds.insert(1);
    /// ds.insert(2);
    /// ds.insert(3);
    ///
    /// assert!(!ds.same_set(&1, &2).unwrap());
    /// assert!(!ds.same_set(&1, &3).unwrap());
    /// assert!(!ds.same_set(&2, &3).unwrap());
    ///
    /// ds.union(&1, &2).unwrap();
    /// ds.union(&2, &3).unwrap();
    ///
    /// assert!(ds.same_set(&1, &2).unwrap());
    /// assert!(ds.same_set(&1, &3).unwrap());
    /// assert!(ds.same_set(&2, &3).unwrap());
    ///
    /// // calling same_set on elements not already contained within the partition
    /// assert!(ds.same_set(&1, &4).is_err());
    /// ```
    pub fn same_set(&mut self, x: &T, y: &T) -> Result<bool, DisjointSetError> {
        let x = *self
            .val_to_index
            .get(x)
            .ok_or(DisjointSetError::MissingElement)?;
        let y = *self
            .val_to_index
            .get(y)
            .ok_or(DisjointSetError::MissingElement)?;
        let sx = self.find_compress(x);
        let sy = self.find_compress(y);

        Ok(sx == sy)
    }

    /// An iterator visiting all elements in arbitrary order.
    /// The iterator element type is &'a T.
    ///
    /// # Examples
    /// ```
    /// use disjoint_set_forest::DisjointSet;
    /// let mut ds = DisjointSet::new();
    /// ds.insert(1);
    /// ds.insert(2);
    /// ds.insert(3);
    ///
    /// // prints 1, 2, 3 in arbitrary order
    /// for x in ds.iter() {
    ///     println!("{}", x);
    /// }
    /// ```
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            hashmap_iter: self.val_to_index.iter(),
        }
    }

    /// An iterator of iterators over the sets in the disjoint set partition
    /// The iterator element type is iterators with element &'a T
    ///
    /// # Examples
    /// ```
    /// use disjoint_set_forest::DisjointSet;
    /// let mut ds = DisjointSet::new();
    /// ds.insert(1);
    /// ds.insert(2);
    /// ds.insert(3);
    /// ds.union(&1, &2).unwrap();
    ///
    /// // prints set: 1 2 set: 3 or with some permutation of the sets or elements within each set
    /// for set in ds.sets() {
    ///     println!("set:");
    ///     for x in set {
    ///         println!("{}", x);
    ///     }
    /// }
    /// ```
    pub fn partition(&self) -> Partition<'_, T> {
        let mut parents = Vec::new();
        for i in 0..self.len() {
            parents.push(self.find(i));
        }

        let mut partition = Vec::new();
        for _ in 0..self.len() {
            partition.push(Vec::new());
        }
        for (element, &index) in self.val_to_index.iter() {
            partition[parents[index]].push(element);
        }

        let mut set_elements = Vec::new();
        for subset in partition {
            if !subset.is_empty() {
                set_elements.push(Subset {
                    subset_iter: subset.into_iter(),
                })
            }
        }
        Partition {
            partition_iter: set_elements.into_iter(),
        }
    }

    /// An owning iterator over the partition of subsets
    pub fn into_partition(self) -> IntoPartition<T> {
        let mut parents = Vec::new();
        for i in 0..self.len() {
            parents.push(self.find(i));
        }

        let mut partition = Vec::new();
        for _ in 0..self.len() {
            partition.push(Vec::new());
        }
        for (element, index) in self.val_to_index.into_iter() {
            partition[parents[index]].push(element);
        }

        let mut set_elements = Vec::new();
        for subset in partition {
            if !subset.is_empty() {
                set_elements.push(IntoSubset {
                    subset_into_iter: subset.into_iter(),
                })
            }
        }
        IntoPartition {
            partition_into_iter: set_elements.into_iter(),
        }
    }

    /// Returns the number of elements the disjoint set data structure can hold without reallocating.
    ///
    /// This number is a lower bound; the [`DisjointSet`] may be able to hold more, but it is guaranteed
    /// to be able to hold at least this many
    ///
    /// # Examples
    /// ```
    /// use disjoint_set_forest::DisjointSet;
    /// let mut ds: DisjointSet<i32> = DisjointSet::with_capacity(100);
    /// 
    /// assert!(ds.capacity() >= 100);
    /// ```
    pub fn capacity(&mut self) -> usize {
        min(
            self.val_to_index.capacity(),
            min(self.parents.capacity(), self.sizes.capacity()),
        )
    }

    /// Reserves capacity for at least `additional` more elements to be inserted in the [`DisjointSet`].
    /// The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Examples
    /// ```
    /// use disjoint_set_forest::DisjointSet;
    /// let mut ds: DisjointSet<i32> = DisjointSet::new();
    /// ds.reserve(100);
    /// 
    /// assert!(ds.capacity() >= 100);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        self.val_to_index.reserve(additional);
        self.parents.reserve(additional);
        self.sizes.reserve(additional);
    }

    /// Shrinks the capacity of the disjoint set data structure as much as possible.
    /// It will drop down as much as possible while maintaining the internal rules
    /// and possibly leaving some space in accordance with the resize policy.
    ///
    /// # Examples
    /// ```
    /// use disjoint_set_forest::DisjointSet;
    /// let mut ds: DisjointSet<i32> = DisjointSet::with_capacity(100);
    /// 
    /// assert!(ds.capacity() >= 100);
    /// 
    /// ds.shrink_to_fit();
    /// 
    /// assert!(ds.capacity() == 0);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.val_to_index.shrink_to_fit();
        self.parents.shrink_to_fit();
        self.sizes.shrink_to_fit();
    }

    /// Clears the disjoint set data structure, removing all elements
    ///
    /// # Examples
    /// ```
    /// use disjoint_set_forest::DisjointSet;
    /// let mut ds = DisjointSet::new();
    /// ds.insert(5);
    /// ds.insert(10);
    /// ds.union(&5, &10);
    /// ds.clear();
    /// 
    /// assert!(ds.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.val_to_index.clear();
        self.parents.clear();
        self.sizes.clear();
    }
}

/// An iterator over the elements of a `DisjointSet`.
///
/// This `struct` is created by the `iter` method on [`DisjointSet`].
/// See its documentation for more.
pub struct Iter<'a, T: Hash + Eq> {
    hashmap_iter: std::collections::hash_map::Iter<'a, T, usize>,
}

impl<'a, T: Hash + Eq> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.hashmap_iter.next().map(|x| x.0)
    }
}

impl<T: Hash + Eq> IntoIterator for DisjointSet<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    fn into_iter(self) -> IntoIter<T> {
        IntoIter {
            hashmap_into_iter: self.val_to_index.into_iter(),
        }
    }
}

/// An owning iterator over the elements of a `DisjointSet`.
///
/// This `struct` is created by the `into_iter` method on [`DisjointSet`].
/// (provided by the `IntoIterator` trait). See its documentation for more.
pub struct IntoIter<T: Hash + Eq> {
    hashmap_into_iter: std::collections::hash_map::IntoIter<T, usize>,
}

impl<T: Hash + Eq> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.hashmap_into_iter.next().map(|x| x.0)
    }
}

/// iterator over partition
pub struct Partition<'a, T: Hash + Eq> {
    partition_iter: std::vec::IntoIter<Subset<'a, T>>,
}

impl<'a, T: Hash + Eq> Iterator for Partition<'a, T> {
    type Item = Subset<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.partition_iter.next()
    }
}

/// iterator over subset
#[derive(Clone)]
pub struct Subset<'a, T: Hash + Eq> {
    subset_iter: std::vec::IntoIter<&'a T>,
}

impl<'a, T: Hash + Eq> Iterator for Subset<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.subset_iter.next()
    }
}

impl<T: fmt::Debug + Hash + Eq + Clone> fmt::Debug for Subset<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.clone()).finish()
    }
}

/// owning iterator over partition
pub struct IntoPartition<T: Hash + Eq> {
    partition_into_iter: std::vec::IntoIter<IntoSubset<T>>,
}

impl<T: Hash + Eq> Iterator for IntoPartition<T> {
    type Item = IntoSubset<T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.partition_into_iter.next()
    }
}

/// owning iterator over subset
pub struct IntoSubset<T: Hash + Eq> {
    subset_into_iter: std::vec::IntoIter<T>,
}

impl<T: Hash + Eq> Iterator for IntoSubset<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.subset_into_iter.next()
    }
}

impl<T: Hash + Eq> FromIterator<T> for DisjointSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut ds = DisjointSet::new();
        for i in iter {
            ds.insert(i);
        }
        ds
    }
}

impl<T: Hash + Eq> Extend<T> for DisjointSet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for i in iter {
            self.insert(i);
        }
    }
}

impl<T: Hash + Eq> PartialEq for DisjointSet<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        let mut permutation = vec![0; self.len()];
        for (key, self_index) in self.val_to_index.iter() {
            match other.val_to_index.get(key) {
                Some(other_index) => permutation[*self_index] = *other_index,
                None => return false,
            }
        }
        let mut mapping = vec![None; self.len()];
        let mut is_mapped_to = vec![false; self.len()];
        for (self_index, other_index) in permutation.into_iter().enumerate() {
            let self_set = self.find(self_index);
            let other_set = other.find(other_index);
            match mapping[self_set] {
                Some(map_self_set) => {
                    if map_self_set != other_set {
                        return false;
                    }
                }
                None => {
                    if is_mapped_to[other_set] {
                        return false;
                    } else {
                        mapping[self_set] = Some(other_set);
                        is_mapped_to[other_set] = true;
                    }
                }
            }
        }
        true
    }
}

impl<T: fmt::Debug + Hash + Eq + Clone> fmt::Debug for DisjointSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.partition()).finish()
    }
}

/// The possible errors that may be raised by the `DisjointSet`.
#[derive(Debug)]
#[non_exhaustive]
pub enum DisjointSetError {
    MissingElement,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_insert() {
        let mut ds = DisjointSet::new();
        assert!(ds.is_empty());
        assert_eq!(ds.len(), 0);

        assert!(ds.insert(1));
        assert!(!ds.insert(1));
        assert!(!ds.insert(1));
        assert!(ds.insert(2));
        assert!(!ds.is_empty());
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn test_err() {
        let mut ds = DisjointSet::new();
        assert!(ds.union(&1, &2).is_err());
        assert!(ds.same_set(&1, &2).is_err());

        assert!(ds.insert(1));
        assert!(ds.union(&1, &2).is_err());
        assert!(ds.same_set(&1, &2).is_err());

        assert!(!ds.insert(1));
        assert!(ds.union(&1, &2).is_err());
        assert!(ds.same_set(&1, &2).is_err());

        assert!(ds.insert(2));
        assert!(ds.union(&1, &2).is_ok());
        assert!(ds.same_set(&1, &2).is_ok());
    }

    #[test]
    fn test_union_and_same_set() {
        let mut ds = DisjointSet::new();
        for i in 0..8 {
            assert!(ds.insert(i));
        }
        assert!(!ds.same_set(&0, &2).unwrap());
        assert!(!ds.same_set(&0, &2).unwrap());
        assert!(!ds.same_set(&4, &0).unwrap());

        ds.union(&2, &4).unwrap();
        assert!(ds.same_set(&2, &4).unwrap());
        assert!(ds.same_set(&4, &2).unwrap());

        ds.union(&4, &2).unwrap();
        assert!(ds.same_set(&2, &4).unwrap());
        assert!(ds.same_set(&4, &2).unwrap());

        ds.union(&2, &6).unwrap();
        assert!(ds.same_set(&2, &6).unwrap());
        assert!(ds.same_set(&6, &4).unwrap());

        ds.union(&0, &7).unwrap();
        ds.union(&5, &0).unwrap();
        assert!(!ds.same_set(&5, &2).unwrap());
        assert!(ds.same_set(&6, &4).unwrap());

        ds.union(&5, &6).unwrap();
        ds.union(&1, &3).unwrap();
        assert!(ds.same_set(&7, &2).unwrap());
        assert!(ds.same_set(&1, &3).unwrap());
        assert!(!ds.same_set(&1, &7).unwrap());
        assert!(!ds.same_set(&3, &0).unwrap());
    }

    #[test]
    fn test_iter() {
        let mut ds = DisjointSet::new();
        for i in 0..10 {
            assert!(ds.insert(i));
        }
        let mut items = Vec::new();
        for x in ds.iter() {
            items.push(*x);
        }
        items.sort_unstable();
        assert_eq!(items, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

        assert!(!ds.insert(2));
        assert!(ds.insert(15));
    }

    #[test]
    fn test_into_iter() {
        let mut ds = DisjointSet::new();
        for i in 0..10 {
            assert!(ds.insert(i));
        }
        let mut items: Vec<u32> = Vec::new();
        for x in ds {
            items.push(x);
        }
        items.sort_unstable();
        assert_eq!(items, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_from_iter() {
        let items = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let ds: DisjointSet<u32> = items.into_iter().collect();
        for i in 0..10 {
            assert!(ds.contains(&i));
        }
        assert!(!ds.contains(&10));
        assert!(!ds.contains(&11));
    }

    #[test]
    fn test_extend() {
        let mut ds = DisjointSet::new();
        assert!(ds.insert(3));
        assert!(ds.insert(12));
        assert!(ds.insert(10));

        let items = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        ds.extend(items.into_iter());
        for i in 0..11 {
            assert!(ds.contains(&i));
        }
        assert!(!ds.contains(&11));
        assert!(ds.contains(&12));
    }

    #[test]
    fn test_eq() {
        let mut ds1 = DisjointSet::new();
        assert!(ds1.insert(3));
        assert!(ds1.insert(12));
        assert!(ds1.insert(10));
        ds1.union(&3, &12).unwrap();

        let mut ds2 = DisjointSet::new();
        assert_ne!(ds1, ds2);
        assert_ne!(ds2, ds1);

        assert!(ds2.insert(3));
        assert_ne!(ds1, ds2);
        assert_ne!(ds2, ds1);

        assert!(ds2.insert(12));
        assert_ne!(ds1, ds2);
        assert_ne!(ds2, ds1);

        assert!(ds2.insert(2));
        assert_ne!(ds1, ds2);
        assert_ne!(ds2, ds1);

        assert!(ds1.insert(2));
        assert!(ds2.insert(10));
        assert_ne!(ds1, ds2);
        assert_ne!(ds2, ds1);

        ds2.union(&3, &12).unwrap();
        assert_eq!(ds1, ds2);
        assert_eq!(ds2, ds1);

        ds2.union(&2, &12).unwrap();
        assert_ne!(ds1, ds2);
        assert_ne!(ds2, ds1);

        ds1.union(&3, &2).unwrap();
        assert_eq!(ds1, ds2);
        assert_eq!(ds2, ds1);
    }

    #[test]
    fn test_partition() {
        let mut ds = DisjointSet::new();
        for i in 0..8 {
            assert!(ds.insert(i));
        }
        ds.union(&2, &4).unwrap();
        ds.union(&4, &2).unwrap();
        ds.union(&1, &7).unwrap();

        let mut partition_vec = Vec::new();
        for subset in ds.partition() {
            let mut curr = Vec::new();
            for element in subset {
                curr.push(*element);
            }
            curr.sort_unstable();
            partition_vec.push(curr);
        }
        partition_vec.sort_unstable();
        assert_eq!(
            partition_vec,
            vec![vec![0], vec![1, 7], vec![2, 4], vec![3], vec![5], vec![6]]
        );

        ds.union(&3, &5).unwrap();
        ds.union(&2, &6).unwrap();

        let mut partition_vec = Vec::new();
        for subset in ds.partition() {
            let mut curr = Vec::new();
            for element in subset {
                curr.push(*element);
            }
            curr.sort_unstable();
            partition_vec.push(curr);
        }
        partition_vec.sort_unstable();
        assert_eq!(
            partition_vec,
            vec![vec![0], vec![1, 7], vec![2, 4, 6], vec![3, 5]]
        );
    }
}
