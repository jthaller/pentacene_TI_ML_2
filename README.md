# pentacene_ML_2
recommended: read pentacene_TI_ML readme.md first.

This program starts with the calculated coulomb potentials between a dimer of two pentacene molecules. Each monomoer contains 30 atoms, so there are 30 distances between each atom of one monomer and all the other atoms in the second monomer. Thus, there are a total of 30**X**30 =900 features. So the *coulbomb_interactions.pic* file allows you to create a dataframe of 10,000 x 900, meaning that there are 10,0000 examples, each with the 900 features. The TI.pic file allows you to create a data frame of size 10,000 x 1, meaning, here are the calculated transport integrals for each of the 10,000 examples.



