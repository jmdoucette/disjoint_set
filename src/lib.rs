use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::iter::FromIterator;

/// A disjoint set implemented using a disjoint set forest.
#[derive(Clone, Default, Eq)]
pub struct DisjointSet<T: Hash + Eq> {
    val_to_index: HashMap<T, usize>,
    parents: Vec<usize>,
    sizes: Vec<usize>,
}

impl<T: Hash + Eq> DisjointSet<T> {
    /// Creates a new, empty `DisjointSet`.
    pub fn new() -> Self {
        Self {
            val_to_index: HashMap::new(),
            parents: Vec::new(),
            sizes: Vec::new(),
        }
    }

    /// Creates a new, empty `DisjointSet` with the specified capacity.
    ///
    /// This preallocates enough memory for `capacity` elements,
    /// so that the `DisjointSet` does not have to be reallocated
    /// until it contains at least that many values.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            val_to_index: HashMap::with_capacity(capacity),
            parents: Vec::with_capacity(capacity),
            sizes: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of elements in a disjoint set data structure.
    pub fn len(&self) -> usize {
        self.val_to_index.len()
    }

    /// Returns `true` if the disjoint set data structure contains no elements.
    pub fn is_empty(&self) -> bool {
        self.val_to_index.is_empty()
    }

    /// Returns `true` if the disjoint set data structure contains the specified element.
    pub fn contains(&self, x: &T) -> bool {
        self.val_to_index.contains_key(x)
    }

    /// Adds a element to the disjoint set data structure as a singleton set.
    ///
    /// If the disjoint set data structure did not contain this element, `true` is returned.
    /// If the disjoint set data structure did contain this element, `false` is returned.
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
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            hashmap_iter: self.val_to_index.iter(),
        }
    }

    pub fn sets(&self) -> Sets<'_, T> {
        let mut index_to_val = HashMap::new();
        for (element, &index) in self.val_to_index.iter() {
            index_to_val.insert(index, element);
        }

        let mut sets = Vec::new();
        for _ in 0..self.len() {
            sets.push(Vec::new());
        }
        for i in 0..self.len() {
            sets[self.find(i)].push(index_to_val[&i]);
        }

        let mut set_elements = Vec::new();
        for set in sets {
            if !set.is_empty() {
                set_elements.push(SetElements {
                    element_iter: set.into_iter(),
                })
            }
        }
        Sets {
            set_iter: set_elements.into_iter(),
        }
    }
}

/// An iterator over the elements of a `DisjointSet`.
///
/// This `struct` is created by the [`iter`] method on [`DisjointSet`].
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
/// This `struct` is created by the [`into_iter`] method on [`DisjointSet`].
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

pub struct Sets<'a, T: Hash + Eq> {
    pub set_iter: std::vec::IntoIter<SetElements<'a, T>>,
}

impl<'a, T: Hash + Eq> Iterator for Sets<'a, T> {
    type Item = SetElements<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.set_iter.next()
    }
}

#[derive(Clone)]
pub struct SetElements<'a, T: Hash + Eq> {
    pub element_iter: std::vec::IntoIter<&'a T>,
}

impl<'a, T: Hash + Eq> Iterator for SetElements<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.element_iter.next()
    }
}

impl<T: fmt::Debug + Hash + Eq + Clone> fmt::Debug for SetElements<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.clone()).finish()
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
        f.debug_set().entries(self.sets()).finish()
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
    fn test_sets() {
        let mut ds = DisjointSet::new();
        for i in 0..8 {
            assert!(ds.insert(i));
        }
        ds.union(&2, &4).unwrap();
        ds.union(&4, &2).unwrap();
        ds.union(&1, &7).unwrap();

        let mut sets_vec = Vec::new();
        for set in ds.sets() {
            let mut curr = Vec::new();
            for element in set {
                curr.push(*element);
            }
            curr.sort_unstable();
            sets_vec.push(curr);
        }
        sets_vec.sort_unstable();
        assert_eq!(
            sets_vec,
            vec![vec![0], vec![1, 7], vec![2, 4], vec![3], vec![5], vec![6]]
        );

        ds.union(&3, &5).unwrap();
        ds.union(&2, &6).unwrap();

        let mut sets_vec = Vec::new();
        for set in ds.sets() {
            let mut curr = Vec::new();
            for element in set {
                curr.push(*element);
            }
            curr.sort_unstable();
            sets_vec.push(curr);
        }
        sets_vec.sort_unstable();
        assert_eq!(
            sets_vec,
            vec![vec![0], vec![1, 7], vec![2, 4, 6], vec![3, 5]]
        );
    }
}
