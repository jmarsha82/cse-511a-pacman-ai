const projects = {
  0: {
    title: "Project 0: Python Warm-Up",
    kicker: "Unit-tested module",
    name: "Fruit shop utilities",
    description: "Introductory Python exercises for ordering fruit, comparing shops, list comprehensions, and quick sort.",
    files: "project0/buyLotsOfFruit.py, shop.py, shopSmart.py",
    tests: "order totals, missing fruit, lowest-cost shop selection",
  },
  1: {
    title: "Project 1: Search",
    kicker: "Unit-tested module",
    name: "Graph search for Pacman mazes",
    description: "Depth-first, breadth-first, uniform-cost, and A* search implementations used by Pacman search agents.",
    files: "project1/search.py, searchAgents.py, util.py",
    tests: "BFS, DFS, UCS, A*, no-solution behavior, Counter helpers",
  },
  2: {
    title: "Project 2: Multi-Agent Search",
    kicker: "Assignment module",
    name: "Adversarial Pacman agents",
    description: "Reflex, minimax, alpha-beta, and expectimax agents for Pacman games with ghosts.",
    files: "project2/multiAgents.py, pacman.py, game.py",
    tests: "covered by compile smoke checks; candidate for deeper game-state fixtures",
  },
  3: {
    title: "Project 3: Reinforcement Learning",
    kicker: "Unit-tested module",
    name: "Value iteration and Q-learning",
    description: "MDP value iteration plus tabular and approximate Q-learning agents for gridworld and Pacman.",
    files: "project3/valueIterationAgents.py, qlearningAgents.py",
    tests: "policy selection, Q-value updates, terminal states, approximate weights",
  },
  4: {
    title: "Project 4: Ghostbusters Inference",
    kicker: "Assignment module",
    name: "Probabilistic ghost tracking",
    description: "Exact inference, particle filtering, and joint particle filtering for noisy ghost observations.",
    files: "project4/inference.py, busters.py, bustersAgents.py",
    tests: "covered by compile smoke checks; candidate for particle-filter fixtures",
  },
};

const buttons = document.querySelectorAll(".project-item");
const nodes = document.querySelectorAll(".node");

function selectProject(id) {
  const project = projects[id];
  document.querySelector("#project-title").textContent = project.title;
  document.querySelector("#project-kicker").textContent = project.kicker;
  document.querySelector("#project-name").textContent = project.name;
  document.querySelector("#project-description").textContent = project.description;
  document.querySelector("#project-files").textContent = project.files;
  document.querySelector("#project-tests").textContent = project.tests;

  buttons.forEach((button) => button.classList.toggle("active", button.dataset.project === id));
  nodes.forEach((node) => node.classList.toggle("active", node.dataset.node === id));
}

buttons.forEach((button) => {
  button.addEventListener("click", () => selectProject(button.dataset.project));
});
