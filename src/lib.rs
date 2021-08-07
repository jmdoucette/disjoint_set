use std::collections::HashMap;
use std::hash::Hash;
use std::iter::FromIterator;

#[derive(Debug)]
#[non_exhaustive]
pub enum DisjointSetError {
    MissingElement,
    Full,
}

#[derive(Clone, Default, Debug, Eq)]
pub struct DisjointSetPathSplitting<T: Hash + Eq> {
    val_to_index: HashMap<T, usize>,
    parents: Vec<Option<usize>>,
    sizes: Vec<usize>,
}

pub struct Iter<'a, T: Hash + Eq> {
    hashmap_iter: std::collections::hash_map::Iter<'a, T, usize>,
}

impl<'a, T: Hash + Eq> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.hashmap_iter.next().map(|x| x.0)
    }
}

pub struct IntoIter<T: Hash + Eq> {
    hashmap_into_iter: std::collections::hash_map::IntoIter<T, usize>,
}

impl<T: Hash + Eq> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.hashmap_into_iter.next().map(|x| x.0)
    }
}

impl<T: Hash + Eq> DisjointSetPathSplitting<T> {
    pub fn new() -> Self {
        Self {
            val_to_index: HashMap::new(),
            parents: Vec::new(),
            sizes: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            val_to_index: HashMap::with_capacity(capacity),
            parents: Vec::with_capacity(capacity),
            sizes: Vec::with_capacity(capacity),
        }
    }

    /// Checks if a disoint set is empty
    pub fn is_empty(&self) -> bool {
        self.val_to_index.is_empty()
    }

    /// Returns the number of elements in a disjoint set
    pub fn len(&self) -> usize {
        self.val_to_index.len()
    }

    pub fn contains(&self, x: &T) -> bool {
        self.val_to_index.contains_key(x)
    }

    /// Adds an element to a disjoint set. Returns true if item was not present, false if it was
    ///
    /// # Arguments
    ///
    /// * `x` - Item to add to disoint set
    pub fn add(&mut self, x: T) -> bool {
        if self.contains(&x) {
            return false;
        }

        self.val_to_index.insert(x, self.len());
        self.parents.push(None);
        self.sizes.push(1);
        true
    }

    /// Combines the sets containing the two given elements
    ///
    /// # Arguments
    ///
    /// * `x` - First items to union
    /// * `y` - Second item to union
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
            // x is in the larger set, x becomes ys parent
            if self.sizes[sx] >= self.sizes[sy] {
                self.parents[sy] = Some(sx);
                self.sizes[sx] += self.sizes[sy];
            }
            // y is in the larger set, y becomes xs parent
            else {
                self.parents[sx] = Some(sy);
                self.sizes[sy] += self.sizes[sx];
            }
        }
        Ok(())
    }

    /// Finds the number of the set containing an element
    ///
    /// # Arguments
    ///
    /// * `x` - Item to find
    fn find(&self, x: usize) -> usize {
        let mut curr = x;
        while let Some(parent) = self.parents[curr] {
            curr = parent;
        }
        curr
    }

    fn find_compress(&mut self, x: usize) -> usize {
        let mut curr = x;
        while let Some(next) = self.parents[curr] {
            curr = next;
            self.parents[curr] = self.parents[next];
        }
        curr
    }

    /// Checks if two elements are in the same set
    ///
    /// # Arguments
    ///
    /// * `x` - First element
    /// * `y` - Second element
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

    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            hashmap_iter: self.val_to_index.iter(),
        }
    }
}

impl<T: Hash + Eq> IntoIterator for DisjointSetPathSplitting<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    fn into_iter(self) -> IntoIter<T> {
        IntoIter {
            hashmap_into_iter: self.val_to_index.into_iter(),
        }
    }
}

impl<T: Hash + Eq> FromIterator<T> for DisjointSetPathSplitting<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut ds = DisjointSetPathSplitting::new();
        for i in iter {
            ds.add(i);
        }
        ds
    }
}

impl<T: Hash + Eq> Extend<T> for DisjointSetPathSplitting<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for i in iter {
            self.add(i);
        }
    }
}

impl<T: Hash + Eq> PartialEq for DisjointSetPathSplitting<T> {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_add() {
        let mut ds = DisjointSetPathSplitting::new();
        assert!(ds.is_empty());
        assert_eq!(ds.len(), 0);

        assert!(ds.add(1));
        assert!(!ds.add(1));
        assert!(!ds.add(1));
        assert!(ds.add(2));
        assert!(!ds.is_empty());
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn test_err() {
        let mut ds = DisjointSetPathSplitting::new();
        assert!(ds.union(&1, &2).is_err());
        assert!(ds.same_set(&1, &2).is_err());

        assert!(ds.add(1));
        assert!(ds.union(&1, &2).is_err());
        assert!(ds.same_set(&1, &2).is_err());

        assert!(!ds.add(1));
        assert!(ds.union(&1, &2).is_err());
        assert!(ds.same_set(&1, &2).is_err());

        assert!(ds.add(2));
        assert!(ds.union(&1, &2).is_ok());
        assert!(ds.same_set(&1, &2).is_ok());
    }

    #[test]
    fn test_union_and_same_set() {
        let mut ds = DisjointSetPathSplitting::new();
        for i in 0..8 {
            assert!(ds.add(i));
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
        let mut ds = DisjointSetPathSplitting::new();
        for i in 0..10 {
            assert!(ds.add(i));
        }
        let mut items = Vec::new();
        for x in ds.iter() {
            items.push(*x);
        }
        items.sort_unstable();
        assert_eq!(items, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

        //ensuring it has not been moved
        assert!(!ds.add(2));
        assert!(ds.add(15));
    }

    #[test]
    fn test_into_iter() {
        let mut ds = DisjointSetPathSplitting::new();
        for i in 0..10 {
            assert!(ds.add(i));
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
        let ds: DisjointSetPathSplitting<u32> = items.into_iter().collect();
        for i in 0..10 {
            assert!(ds.contains(&i));
        }
        assert!(!ds.contains(&10));
        assert!(!ds.contains(&11));
    }

    #[test]
    fn test_extend() {
        let mut ds = DisjointSetPathSplitting::new();
        assert!(ds.add(3));
        assert!(ds.add(12));
        assert!(ds.add(10));

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
        let mut ds1 = DisjointSetPathSplitting::new();
        assert!(ds1.add(3));
        assert!(ds1.add(12));
        assert!(ds1.add(10));
        ds1.union(&3, &12).unwrap();

        let mut ds2 = DisjointSetPathSplitting::new();
        assert_ne!(ds1, ds2);
        assert_ne!(ds2, ds1);

        assert!(ds2.add(3));
        assert_ne!(ds1, ds2);
        assert_ne!(ds2, ds1);

        assert!(ds2.add(12));
        assert_ne!(ds1, ds2);
        assert_ne!(ds2, ds1);

        assert!(ds2.add(2));
        assert_ne!(ds1, ds2);
        assert_ne!(ds2, ds1);

        assert!(ds1.add(2));
        assert!(ds2.add(10));
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
}
