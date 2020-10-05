<img align="right" width="100" height="100" src="https://github.com/cs-MohamedAyman/Coursera-Specializations/blob/master/organizations-logos/imperial%20college%20london.jpg">

# [Mathematics for Machine Learning Specialization](https://www.coursera.org/specializations/mathematics-machine-learning) `60H`

## WHAT YOU WILL LEARN
- Implement mathematical concepts using real-world data
- Derive PCA from a projection perspective
- Understand how orthogonal projections work
- Master PCA

## SKILLS YOU WILL GAIN
`eigenvalues and eigenvectors` `principal component analysis (pca)` `multivariable calculus` `linear algebra` `basis (linear algebra)` `transformation matrix` `linear regression` `vector calculus` `gradient descent` `dimensionality reduction` `python programming`

## About this Specialization
- For a lot of higher level courses in Machine Learning and Data Science, you find you need to freshen up on the basics in mathematics - stuff you may have studied before in school or university, but which was taught in another context, or not very intuitively, such that you struggle to relate it to how it’s used in Computer Science. This specialization aims to bridge that gap, getting you up to speed in the underlying mathematics, building an intuitive understanding, and relating it to Machine Learning and Data Science.

- In the first course on Linear Algebra we look at what linear algebra is and how it relates to data. Then we look through what vectors and matrices are and how to work with them.

- The second course, Multivariate Calculus, builds on this to look at how to optimize fitting functions to get good fits to data. It starts from introductory calculus and then uses the matrices and vectors from the first course to look at data fitting.

- The third course, Dimensionality Reduction with Principal Component Analysis, uses the mathematics from the first two courses to compress high-dimensional data. This course is of intermediate difficulty and will require Python and numpy knowledge.

- At the end of this specialization you will have gained the prerequisite mathematical knowledge to continue your journey and take more advanced courses in machine learning.

## Applied Learning Project
Through the assignments of this specialisation you will use the skills you have learned to produce mini-projects with Python on interactive notebooks, an easy to learn tool which will help you apply the knowledge to real world problems. For example, using linear algebra in order to calculate the page rank of a small simulated internet, applying multivariate calculus in order to train your own neural network, performing a non-linear least squares regression to fit a model to a data set, and using principal component analysis to determine the features of the MNIST digits data set.

<details>
	<summary>Specialization Details</summary>

- In the first course on Linear Algebra we look at what linear algebra is and how it relates to vectors and matrices. Then we look through what vectors and matrices are and how to work with them, including the knotty problem of eigenvalues and eigenvectors, and how to use these to solve problems. Finally we look at how to use these to do fun things with datasets - like how to rotate images of faces and how to extract eigenvectors to look at how the Pagerank algorithm works.

- Since we're aiming at data-driven applications, we'll be implementing some of these ideas in code, not just on pencil and paper. Towards the end of the course, you'll write code blocks and encounter Jupyter notebooks in Python, but don't worry, these will be quite short, focussed on the concepts, and will guide you through if you’ve not coded before. At the end of this course you will have an intuitive understanding of vectors and matrices that will help you bridge the gap into linear algebra problems, and how to apply these concepts to machine learning.

- The second course offers a brief introduction to the multivariate calculus required to build many common machine learning techniques. We start at the very beginning with a refresher on the “rise over run” formulation of a slope, before converting this to the formal definition of the gradient of a function. We then start to build up a set of tools for making calculus easier and faster. Next, we learn how to calculate vectors that point up hill on multidimensional surfaces and even put this into action using an interactive game. We take a look at how we can use calculus to build approximations to functions, as well as helping us to quantify how accurate we should expect those approximations to be. We also spend some time talking about where calculus comes up in the training of neural networks, before finally showing you how it is applied in linear regression models. This course is intended to offer an intuitive understanding of calculus, as well as the language necessary to look concepts up yourselves when you get stuck. Hopefully, without going into too much detail, you’ll still come away with the confidence to dive into some more focused machine learning courses in future.

- This intermediate-level course introduces the mathematical foundations to derive Principal Component Analysis (PCA), a fundamental dimensionality reduction technique. We'll cover some basic statistics of data sets, such as mean values and variances, we'll compute distances and angles between vectors using inner products and derive orthogonal projections of data onto lower-dimensional subspaces. Using all these tools, we'll then derive PCA as a method that minimizes the average squared reconstruction error between data points and their reconstruction.

- At the end of this course, you'll be familiar with important mathematical concepts and you can implement PCA all by yourself. If you’re struggling, you'll find a set of jupyter notebooks that will allow you to explore properties of the techniques and walk you through what you need to do to get on track. If you are already an expert, this course may refresh some of your knowledge. The lectures, examples and exercises require: 
  1. Some ability of abstract thinking 
  2. Good background in linear algebra (e.g., matrix and vector algebra, linear independence, basis) 
  3. Basic background in multivariate calculus (e.g., partial derivatives, basic optimization) 
  4. Basic knowledge in python programming and numpy Disclaimer
- This course is substantially more abstract and requires more programming than the other two courses of the specialization. However, this type of abstract thinking, algebraic manipulation and programming is necessary if you want to understand and develop machine learning algorithms.

</details>

## There are 3 Courses in this Specialization

## Course 1: [Mathematics for Machine Learning: Linear Algebra](https://www.coursera.org/learn/linear-algebra-machine-learning) `20H`

### Week 1: Introduction to Linear Algebra and to Mathematics for Machine Learning
```In this first module we look at how linear algebra is relevant to machine learning and data science. Then we'll wind up the module with an initial introduction to vectors. Throughout, we're focussing on developing your mathematical intuition, not of crunching through algebra or doing long pen-and-paper examples. For many of these operations, there are callable functions in Python that can do the adding up - the point is to appreciate what they do and how they work so that, when things go wrong or there are special cases, you can understand why and what to do.```

<details>
      <summary>Week Details</summary>
<br>

- Welcome to this course
  - Video: Introduction: Solving data science challenges with mathematics
  - Reading: About Imperial College & the team
  - Reading: How to be successful in this course
  - Reading: Grading policy
  - Reading: Additional readings & helpful references
  - Complete our short pre-course survey
- The relationship between machine learning, linear algebra, and vectors and matrices
  - Video: Motivations for linear algebra
  - Video: Getting a handle on vectors
  - Practice Quiz: Exploring parameter space
  - Practice Quiz: Solving some simultaneous equations
- Vectors
  - Video: Operations with vectors
  - Practice Quiz: Doing some vector operations
- Summary
  - Video: Summary
</details>

### Week 2: Vectors are objects that move around space
```In this module, we look at operations we can do with vectors - finding the modulus (size), angle between vectors (dot or inner product) and projections of one vector onto another. We can then examine how the entries describing a vector will depend on what vectors we use to define the axes - the basis. That will then let us determine whether a proposed set of basis vectors are what's called 'linearly independent.' This will complete our examination of vectors, allowing us to move on to matrices in module 3 and then start to solve linear algebra problems.```

<details>
      <summary>Week Details</summary>
<br>

- Introduction
  - Video: Introduction to module 2 - Vectors
- Finding the size of a vector, its angle, and projection
  - Video: Modulus & inner product
  - Video: Cosine & dot product
  - Video: Projection
  - Practice Quiz: Dot product of vectors
- Changing the reference frame
  - Video: Changing basis
  - Practice Quiz: Changing basis
  - Video: Basis, vector space, and linear independence
  - Video: Applications of changing basis
  - Practice Quiz: Linear dependency of a set of vectors
- Doing some real-world vectors examples
  - Quiz: Vector operations assessment
  - Video: Summary
</details>

### Week 3: Matrices in Linear Algebra: Objects that operate on Vectors
```Now that we've looked at vectors, we can turn to matrices. First we look at how to use matrices as tools to solve linear algebra problems, and as objects that transform vectors. Then we look at how to solve systems of linear equations using matrices, which will then take us on to look at inverse matrices and determinants, and to think about what the determinant really is, intuitively speaking. Finally, we'll look at cases of special matrices that mean that the determinant is zero or where the matrix isn't invertible - cases where algorithms that need to invert a matrix will fail.```

<details>
      <summary>Week Details</summary>
<br>

- Introduction to matrices
  - Video: Matrices, vectors, and solving simultaneous equation problems
- Matrices in linear algebra: operating on vectors
  - Video: How matrices transform space
  - Video: Types of matrix transformation
  - Video: Composition or combination of matrix transformations
  - Practice Quiz: Using matrices to make transformations
- Matrix Inverses
  - Video: Solving the apples and bananas problem: Gaussian elimination
  - Video: Going from Gaussian elimination to finding the inverse matrix
  - Practice Quiz: Solving linear equations using the inverse matrix
- Special matrices and Coding up some matrix operations
  - Video: Determinants and inverses
  - Lab: Identifying special matrices
  - Video: Summary
  - Programming Assignment: Identifying special matrices
</details>

### Week 4: Matrices make linear mappings
```In Module 4, we continue our discussion of matrices; first we think about how to code up matrix multiplication and matrix operations using the Einstein Summation Convention, which is a widely used notation in more advanced linear algebra courses. Then, we look at how matrices can transform a description of a vector from one basis (set of axes) to another. This will allow us to, for example, figure out how to apply a reflection to an image and manipulate images. We'll also look at how to construct a convenient basis vector set in order to do such transformations. Then, we'll write some code to do these transformations and apply this work computationally.```

<details>
      <summary>Week Details</summary>
<br>

- Matrices as objects that map one vector onto another; all the types of matrices
  - Video: Introduction: Einstein summation convention and the symmetry of the dot product
  - Practice Quiz: Non-square matrix multiplication
  - Practice Quiz: Example: Using non-square matrices to do a projection
- Matrices transform into the new basis vector set
  - Video: Matrices changing basis
  - Video: Doing a transformation in a changed basis
- Making Multiple Mappings, deciding if these are reversible
  - Video: Orthogonal matrices
- Recognising mapping matrices and applying these to data
  - Video: The Gram–Schmidt process
  - Lab: Gram-Schmidt process
  - Video: Example: Reflecting in a plane
  - Lab: Reflecting Bear
  - Programming Assignment: Gram-Schmidt Process
  - Programming Assignment: Reflecting Bear
</details>

### Week 5: Eigenvalues and Eigenvectors: Application to Data Problems
```Eigenvectors are particular vectors that are unrotated by a transformation matrix, and eigenvalues are the amount by which the eigenvectors are stretched. These special 'eigen-things' are very useful in linear algebra and will let us examine Google's famous PageRank algorithm for presenting web search results. Then we'll apply this in code, which will wrap up the course.```

<details>
      <summary>Week Details</summary>
<br>

- What are eigen-things?
  - Video: Welcome to module 5
  - Video: What are eigenvalues and eigenvectors?
  - Practice Quiz: Selecting eigenvectors by inspection
- Getting into the detail of eigenproblems
  - Video: Special eigen-cases
  - Video: Calculating eigenvectors
  - Practice Quiz: Characteristic polynomials, eigenvalues and eigenvectors
- When changing to the eigenbasis is really useful
  - Video: Changing to the eigenbasis
  - Video: Eigenbasis example
  - Practice Quiz: Diagonalisation and applications
  - Visualising Matrices and Eigen
- Making the PageRank algorithm
  - Video: Introduction to PageRank
  - Lab: PageRank
  - Programming Assignment: Page Rank
- Eigenvalues and Eigenvectors: Assessment
  - Quiz: Eigenvalues and eigenvectors
  - Video: Summary
  - Video: Wrap up of this linear algebra course
  - Reading: Did you like the course? Let us know!
  - Post-Course Survey
</details>

## Course 2: [Mathematics for Machine Learning: Multivariate Calculus](https://www.coursera.org/learn/multivariate-calculus-machine-learning) `20H`

### Week 1: What is calculus?
```Understanding calculus is central to understanding machine learning! You can think of calculus as simply a set of tools for analysing the relationship between functions and their inputs. Often, in machine learning, we are trying to find the inputs which enable a function to best match the data. We start this module from the basics, by recalling what a function is and where we might encounter one. Following this, we talk about the how, when sketching a function on a graph, the slope describes the rate of change of the output with respect to an input. Using this visual intuition we next derive a robust mathematical definition of a derivative, which we then use to differentiate some interesting functions. Finally, by studying a few examples, we develop four handy time saving rules that enable us to speed up differentiation for many common scenarios.```

<details>
      <summary>Week Details</summary>
<br>

- Welcome to this course
  - Video: Welcome to Multivariate Calculus
  - Reading: About Imperial College & the team
  - Reading: How to be successful in this course
  - Reading: Grading Policy
  - Reading: Additional Readings & Helpful References
  - Pre-course Survey
- Back to basics: functions
  - Video: Welcome to Module 1!
  - Video: Functions
  - Practice Quiz: Matching functions visually
- Gradients and derivatives
  - Video: Rise Over Run
  - Practice Quiz: Matching the graph of a function to the graph of its derivative
  - Video: Definition of a derivative
  - Video: Differentiation examples & special cases
  - Practice Quiz: Let's differentiate some functions
- Time saving rules
  - Video: Product rule
  - Practice Quiz: Practicing the product rule
  - Video: Chain rule
  - Video: Taming a beast
  - Practice Quiz: Practicing the chain rule
- Assessment
  - Quiz: Unleashing the toolbox
  - Video: See you next module!
</details>

### Week 2: Multivariate calculus
```Building on the foundations of the previous module, we now generalise our calculus tools to handle multivariable systems. This means we can take a function with multiple inputs and determine the influence of each of them separately. It would not be unusual for a machine learning method to require the analysis of a function with thousands of inputs, so we will also introduce the linear algebra structures necessary for storing the results of our multivariate calculus analysis in an orderly fashion.```

<details>
      <summary>Week Details</summary>
<br>

- Moving to multivariate
  - Video: Welcome to Module 2!
  - Video: Variables, constants & context
  - Video: Differentiate with respect to anything
  - Practice Quiz: Practicing partial differentiation
- Jacobians - vectors of derivatives
  - Video: The Jacobian
  - Practice Quiz: Calculating the Jacobian
  - Video: Jacobian applied
  - Practice Quiz: Bigger Jacobians!
- The sandpit game
  - Video: The Sandpit
  - Lab: The Sandpit
  - Lab: The Sandpit - Part 2
  - Video: The Hessian
  - Practice Quiz: Calculating Hessians
  - Video: Reality is hard
  - Quiz: Assessment: Jacobians and Hessians
  - Video: See you next module!
</details>

### Week 3: Multivariate chain rule and its applications
```Having seen that multivariate calculus is really no more complicated than the univariate case, we now focus on applications of the chain rule. Neural networks are one of the most popular and successful conceptual structures in machine learning. They are build up from a connected web of neurons and inspired by the structure of biological brains. The behaviour of each neuron is influenced by a set of control parameters, each of which needs to be optimised to best fit the data. The multivariate chain rule can be used to calculate the influence of each parameter of the networks, allow them to be updated during training.```

<details>
      <summary>Week Details</summary>
<br>

- Chain rule intro.
  - Video: Welcome to Module 3!
  - Video: Multivariate chain rule
  - Video: More multivariate chain rule
  - Practice Quiz: Multivariate chain rule exercise
- Neural Networks
  - Video: Simple neural networks
  - Practice Quiz: Simple Artificial Neural Networks
  - Video: More simple neural networks
  - Practice Quiz: Training Neural Networks
  - Lab: Backpropagation
  - Programming Assignment: Backpropagation
  - Video: See you next module!
</details>

### Week 4: Taylor series and linearisation
```The Taylor series is a method for re-expressing functions as polynomial series. This approach is the rational behind the use of simple linear approximations to complicated functions. In this module, we will derive the formal expression for the univariate Taylor series and discuss some important consequences of this result relevant to machine learning. Finally, we will discuss the multivariate case and see how the Jacobian and the Hessian come in to play.```

<details>
      <summary>Week Details</summary>
<br>

- Taylor series for approximations
  - Video: Welcome to Module 4!
  - Video: Building approximate functions
  - Video: Power series
  - Practice Quiz: Matching functions and approximations
  - Visualising Taylor Series
  - Video: Power series derivation
  - Video: Power series details
  - Practice Quiz: Applying the Taylor series
  - Video: Examples
- Multivariable Taylor Series
  - Video: Linearisation
  - Practice Quiz: Taylor series - Special cases
  - Video: Multivariate Taylor
  - Practice Quiz: 2D Taylor series
  - Quiz: Taylor Series Assessment
  - Video: See you next module!
</details>

### Week 5: Intro to optimisation
```If we want to find the minimum and maximum points of a function then we can use multivariate calculus to do this, say to optimise the parameters (the space) of a function to fit some data. First we’ll do this in one dimension and use the gradient to give us estimates of where the zero points of that function are, and then iterate in the Newton-Raphson method. Then we’ll extend the idea to multiple dimensions by finding the gradient vector, Grad, which is the vector of the Jacobian. This will then let us find our way to the minima and maxima in what is called the gradient descent method. We’ll then take a moment to use Grad to find the minima and maxima along a constraint in the space, which is the Lagrange multipliers method.```

<details>
      <summary>Week Details</summary>
<br>

- Fitting as minimisation problem
  - Video: Welcome to Module 5!
  - Practice Quiz: Newton-Raphson in one dimension
  - Video: Gradient Descent
  - Lab: Gradient descent in a sandpit
  - Quiz: Checking Newton-Raphson
- Lagrange multipliers
  - Video: Constrained optimisation
  - Practice Quiz: Lagrange multipliers
- Assessment
  - Quiz: Optimisation scenarios
  - Video: See you next module!
</details>

### Week 6: Regression
```In order to optimise the fitting parameters of a fitting function to the best fit for some data, we need a way to define how good our fit is. This goodness of fit is called chi-squared, which we’ll first apply to fitting a straight line - linear regression. Then we’ll look at how to optimise our fitting function using chi-squared in the general case using the gradient descent method. Finally, we’ll look at how to do this easily in Python in just a few lines of code, which will wrap up the course.```

<details>
      <summary>Week Details</summary>
<br>

- Into to linear regression
  - Video: Simple linear regression
  - Practice Quiz: Linear regression
- Non-linear regression
  - Video: General non linear least squares
  - Practice Quiz: Fitting a non-linear function
  - Video: Doing least squares regression analysis in practice
  - Lab: Fitting the distribution of heights data
  - Programming Assignment: Fitting the distribution of height data
  - Video: Wrap up of this course
  - Reading: Did you like the course? Let us know!
  - Post-course Survey
</details>

## Course 3: [Mathematics for Machine Learning: PCA](https://www.coursera.org/learn/pca-machine-learning) `20H`

### Week 1: Statistics of Datasets
```Principal Component Analysis (PCA) is one of the most important dimensionality reduction algorithms in machine learning. In this course, we lay the mathematical foundations to derive and understand PCA from a geometric point of view. In this module, we learn how to summarize datasets (e.g., images) using basic statistics, such as the mean and the variance. We also look at properties of the mean and the variance when we shift or scale the original data set. We will provide mathematical intuition as well as the skills to derive the results. We will also implement our results in code (jupyter notebooks), which will allow us to practice our mathematical understand to compute averages of image data sets.```

<details>
      <summary>Week Details</summary>
<br>

- Welcome to this course
  - Video: Introduction to the course
  - Reading: About Imperial College & the team
  - Reading: How to be successful in this course
  - Reading: Grading policy
  - Reading: Additional readings & helpful references
  - Survey
  - Reading: Mini numpy tutorial
  - Reading: Set up Jupyter notebook environment offline
- Mean values
  - Video: Welcome to module 1
  - Video: Mean of a dataset
  - Practice Quiz: Mean of datasets
- Variances and covariances
  - Video: Variance of one-dimensional datasets
  - Quiz: Variance of 1D datasets
  - Reading: Symmetric, positive definite matrices
  - Video: Variance of higher-dimensional datasets
  - Practice Quiz: Covariance matrix of a two-dimensional dataset
- Linear transformation of datasets
  - Video: Effect on the mean
  - Video: Effect on the (co)variance
  - Lab: Numpy Tutorial
  - Lab: Mean/covariance of a dataset + effect of a linear transformation
  - Programming Assignment: Mean/covariance of a dataset + effect of a linear transformation
  - Video: See you next module!
</details>

### Week 2: Inner Products
```Data can be interpreted as vectors. Vectors allow us to talk about geometric concepts, such as lengths, distances and angles to characterise similarity between vectors. This will become important later in the course when we discuss PCA. In this module, we will introduce and practice the concept of an inner product. Inner products allow us to talk about geometric concepts in vector spaces. More specifically, we will start with the dot product (which we may still know from school) as a special case of an inner product, and then move toward a more general concept of an inner product, which play an integral part in some areas of machine learning, such as kernel machines (this includes support vector machines and Gaussian processes). We have a lot of exercises in this module to practice and understand the concept of inner products.```

<details>
      <summary>Week Details</summary>
<br>

- Dot product
  - Video: Welcome to module 2
  - Video: Dot product
  - Practice Quiz: Dot product
- Inner products
  - Video: Inner product: definition
  - Quiz: Properties of inner products
  - Video: Inner product: length of vectors
  - Video: Inner product: distances between vectors
  - Practice Quiz: General inner products: lengths and distances
  - Reading: Basis vectors
  - Video: Inner product: angles and orthogonality
  - Quiz: Angles between vectors using a non-standard inner product
  - Lab: Inner products and angles
  - Programming Assignment: Inner products and angles
  - Video: Inner products of functions and random variables (optional)
  - Video: Heading for the next module!
</details>

### Week 3: Orthogonal Projections
```In this module, we will look at orthogonal projections of vectors, which live in a high-dimensional vector space, onto lower-dimensional subspaces. This will play an important role in the next module when we derive PCA. We will start off with a geometric motivation of what an orthogonal projection is and work our way through the corresponding derivation. We will end up with a single equation that allows us to project any vector onto a lower-dimensional subspace. However, we will also understand how this equation came about. As in the other modules, we will have both pen-and-paper practice and a small programming example with a jupyter notebook.```

<details>
      <summary>Week Details</summary>
<br>

- Projections
  - Video: Welcome to module 3
  - Video: Projection onto 1D subspaces
  - Video: Example: projection onto 1D subspaces
  - Quiz: Projection onto a 1-dimensional subspace
  - Video: Projections onto higher-dimensional subspaces
  - Reading: Full derivation of the projection
  - Video: Example: projection onto a 2D subspace
  - Practice Quiz: Project 3D data onto a 2D subspace
  - Lab: Orthogonal projections
  - Programming Assignment: Orthogonal projections
  - Video: This was module 3!
</details>

### Week 4: Principal Component Analysis
```We can think of dimensionality reduction as a way of compressing data with some loss, similar to jpg or mp3. Principal Component Analysis (PCA) is one of the most fundamental dimensionality reduction techniques that are used in machine learning. In this module, we use the results from the first three modules of this course and derive PCA from a geometric point of view. Within this course, this module is the most challenging one, and we will go through an explicit derivation of PCA plus some coding exercises that will make us a proficient user of PCA.```

<details>
      <summary>Week Details</summary>
<br>

- PCA derivation
  - Video: Welcome to module 4
  - Reading: Vector spaces
  - Reading: Orthogonal complements
  - Video: Problem setting and PCA objective
  - Reading: Multivariate chain rule
  - Practice Quiz: Chain rule practice
  - Video: Finding the coordinates of the projected data
  - Video: Reformulation of the objective
  - Reading: Lagrange multipliers
  - Video: Finding the basis vectors that span the principal subspace
- PCA algorithm
  - Video: Steps of PCA
  - Video: PCA in high dimensions
  - Lab: Principal Components Analysis (PCA)
  - Programming Assignment: PCA
  - Video: Other interpretations of PCA (optional)
  - Video: Summary of this module
  - Video: This was the course on PCA
  - Reading: Did you like the course? Let us know!
  - Post-Course Survey
</details>
