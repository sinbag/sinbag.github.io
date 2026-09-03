---
layout: default
title: Research
permalink: /
description: Research and publications by Gurprit Singh on Monte Carlo sampling, physically based rendering, optimization, and generative AI.
---

<div class="about-page">
<style>
.about-page {
  max-width: 980px;
  margin: 0 auto;
  line-height: 1.65;
  color: #202124;
}
.opening-quote {
  margin: 1.5rem 0 0;
  padding: 0 0 0 1rem;
  border-left: 2px solid #8ca5bc;
  color: #536273;
  font-size: 0.95rem;
}
.opening-quote p {
  margin: 0;
  font-style: italic;
}
.opening-quote cite {
  display: block;
  margin-top: 0.25rem;
  color: #778391;
  font-size: 0.88rem;
  font-style: normal;
}
.about-hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 210px;
  gap: 3rem;
  align-items: center;
  margin: 0 0 2rem;
}
.about-hero h1 {
  margin-top: 0;
  margin-bottom: 0.5rem;
  font-size: clamp(1.8rem, 4vw, 2.4rem);
  letter-spacing: -0.025em;
  line-height: 1.15;
}
.about-lead {
  max-width: 680px;
  font-size: 1.08rem;
  color: #4b5563;
  margin: 0 0 1.25rem;
}
.profile-image {
  width: 210px;
  height: 250px;
  object-fit: cover;
  object-position: center top;
  border-radius: 12px;
  border: 1px solid #e1e5ea;
  background: #f6f7f8;
}
.section-nav {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  padding: 0.85rem 0;
  margin: 1rem 0 2rem;
  border-top: 1px solid #e8e8e8;
  border-bottom: 1px solid #e8e8e8;
}
.section-nav a {
  color: #27364a;
  text-decoration: none;
  padding: 0.25rem 0.65rem;
  border-radius: 999px;
}
.section-nav a:hover {
  background: #eef3f8;
}
.about-section {
  margin: 2.75rem 0;
}
.about-section h2 {
  margin-bottom: 0.75rem;
  font-size: 1.65rem;
  letter-spacing: -0.025em;
}
.publication-year {
  margin: 2.5rem 0 0;
  padding: 0 0 0.65rem;
  border-bottom: 2px solid #27364a;
  font-size: 1.1rem;
}
.pub-card {
  display: grid;
  grid-template-columns: 180px minmax(0, 1fr);
  gap: 1.4rem;
  padding: 1.35rem 0;
  border-bottom: 1px solid #e7eaee;
  align-items: start;
}
.pub-thumb {
  width: 180px;
  aspect-ratio: 4 / 3;
  border: 1px solid #e1e5ea;
  border-radius: 8px;
  background-color: #f5f7f9;
  background-size: cover;
  background-repeat: no-repeat;
  background-position: center;
}
.pub-thumb--empty {
  display: grid;
  place-items: center;
  background:
    radial-gradient(circle at 28% 30%, rgba(63, 95, 132, 0.22) 0 2px, transparent 3px),
    radial-gradient(circle at 68% 64%, rgba(63, 95, 132, 0.18) 0 2px, transparent 3px),
    #f3f6f9;
  background-size: 28px 28px, 34px 34px, auto;
}
.pub-thumb--empty span {
  display: grid;
  place-items: center;
  width: 44px;
  height: 44px;
  border: 1px solid #c9d2dc;
  border-radius: 50%;
  color: #53687d;
  background: rgba(255, 255, 255, 0.88);
  font-size: 0.78rem;
  letter-spacing: 0.08em;
}
.pub-title {
  font-weight: 700;
  margin-bottom: 0.3rem;
  color: #162333;
  font-size: 1.04rem;
  line-height: 1.35;
}
.pub-authors,
.pub-venue,
.pub-description,
.pub-note {
  margin: 0.15rem 0;
}
.pub-authors {
  color: #444;
}
.pub-venue {
  color: #666;
  font-size: 0.95rem;
}
.pub-description {
  color: #333;
  font-size: 0.95rem;
}
.pub-note {
  color: #666;
  font-size: 0.9rem;
}
.pub-links {
  margin-top: 0.35rem;
  font-size: 0.95rem;
}
.pub-links a {
  display: inline-block;
  margin: 0.2rem 0.35rem 0.15rem 0;
  padding: 0.14rem 0.5rem;
  border: 1px solid #d7dfe7;
  border-radius: 999px;
  color: #315c88;
  text-decoration: none;
  white-space: nowrap;
}
.pub-links a:hover {
  border-color: #8ca5bc;
  background: #f3f7fa;
}
@media (max-width: 700px) {
  .about-hero {
    grid-template-columns: 1fr;
  }
  .profile-image {
    width: 160px;
    height: 190px;
  }
  .pub-card {
    grid-template-columns: 120px minmax(0, 1fr);
    gap: 1rem;
  }
  .pub-thumb {
    width: 120px;
  }
}
@media (max-width: 480px) {
  .pub-card {
    grid-template-columns: 1fr;
  }
  .pub-thumb {
    width: min(100%, 220px);
  }
}
</style>

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Person",
  "name": "Gurprit Singh",
  "url": "https://sinbag.github.io/",
  "sameAs": [
    "https://github.com/sinbag",
    "https://www.linkedin.com/in/sinbag/"
  ],
  "jobTitle": "Researcher",
  "knowsAbout": [
    "Monte Carlo sampling",
    "Physically based rendering",
    "Optimization",
    "Generative AI"
  ]
}
</script>

<section id="about" class="about-hero">
<div>
<h1>Gurprit Singh</h1>
<p class="about-lead">Researcher working at the intersection of Monte Carlo sampling, rendering, optimization, and generative AI.</p>
<p>For more than a decade, I have developed mathematical models to understand the role of randomness. I am fascinated by noise: currently, by the pivotal role it plays in generative AI, and previously by how it affects the convergence of physically based light transport.</p>
<p>I am from Jalandhar, Punjab, India. After attending Kendriya Vidyalaya No. 2 in Jalandhar, I studied at IIT Delhi.</p>
<p>At the core of my research, I develop Monte Carlo sampling strategies for high-dimensional numerical integration. I am equally interested in Monte Carlo, Quasi-Monte Carlo, and MCMC methods for generative modeling.</p>
<blockquote class="opening-quote">
  <p>The art of noise should not limit itself to an imitative reproduction</p>
  <cite>— Luigi Russolo</cite>
</blockquote>
</div>
<img class="profile-image" src="{{ '/assets/images/profile.png' | relative_url }}" alt="Portrait of Gurprit Singh">
</section>

<nav class="section-nav" aria-label="Section navigation">
<a href="#about">About</a>
<a href="#research">Research</a>
<a href="#publications">Publications</a>
<a href="#contact">Contact</a>
</nav>

<section id="research" class="about-section">
<h2>Research</h2>
<p>My research focuses on Monte Carlo, Quasi-Monte Carlo, and MCMC methods for high-dimensional numerical integration, physically based light transport, inverse rendering, optimization, and generative modeling.</p>
</section>

<section id="publications" class="about-section">
<h2>Publications</h2>
<h3 class="publication-year">2026</h3>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/rao-blackwellized-mcmc-light-transport.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Rao-Blackwellized Markov chain Monte Carlo Light Transport</div>
    <p class="pub-authors">Sascha Holl, Gurprit Singh, Hans-Peter Seidel</p>
    <p class="pub-venue">SIGGRAPH North America 2026</p>
    <p class="pub-description">Q: How can Rao-Blackwellization substantially reduce variance and accelerate convergence in MCMC light transport?</p>
    <p class="pub-links"><a href="https://arxiv.org/abs/2605.09117">arXiv</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/diffusion-restore-real-time-mcmc-light-transport.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Diffusion Restore: Real-Time Markov Chain Monte Carlo Light Transport</div>
    <p class="pub-authors">Sascha Holl, Gurprit Singh, Hans-Peter Seidel</p>
    <p class="pub-venue">SIGGRAPH Asia 2026</p>
    <p class="pub-description">Q: Can nonreversible diffusion dynamics make MCMC light transport both state of the art and real-time?</p>
    <p class="pub-links"><a href="https://arxiv.org/abs/2605.08916">arXiv</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/score-based-generative-modeling-anisotropic-spdes.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Score-Based Generative Modeling through Anisotropic Stochastic Partial Differential Equations</div>
    <p class="pub-authors">Sascha Holl, Jente Vandersanden, Gurprit Singh, Hans-Peter Seidel</p>
    <p class="pub-venue">arXiv, 2026</p>
    <p class="pub-description">Q: Can anisotropic diffusion preserve geometric structure longer and improve score-based image generation?</p>
    <p class="pub-links"><a href="https://arxiv.org/abs/2605.08976">arXiv</a></p>
  </div>
</article>
<h3 class="publication-year">2025</h3>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/jump-restore-light-transport.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Jump restore light transport</div>
    <p class="pub-authors">Sascha Holl, Gurprit Singh, Hans-Peter Seidel</p>
    <p class="pub-venue">SIGGRAPH Asia 2025</p>
    <p class="pub-links"><a href="https://restore-light-transport.mpi-inf.mpg.de/">project page</a> / <a href="https://arxiv.org/abs/2409.07148">arXiv</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/gaussian-integral-linear-operators-for-precomputed-graphics.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Gaussian integral linear operators for precomputed graphics</div>
    <p class="pub-authors">Haolin Lu, Yash Belhe, Gurprit Singh, Tzu-Mao Li, Toshiya Hachisuka</p>
    <p class="pub-venue">SIGGRAPH Asia 2025</p>
    <p class="pub-links"><a href="https://suikasibyl.github.io/gilo/#/">project page</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/histogram-stratification-for-spatio-temporal-reservoir-sampling.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Histogram Stratification for Spatio-Temporal Reservoir Sampling</div>
    <p class="pub-authors">Corentin Salaün, Martin Bálint, Laurent Belcour, Eric Heitz, Gurprit Singh, Karol Myszkowski</p>
    <p class="pub-venue">SIGGRAPH North America 2025</p>
    <p class="pub-links"><a href="https://iribis.github.io/publication/2025_Stratified_Histogram_Resampling">project page</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/demystifying-noise-role-of-randomness-in-generative-ai.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Demystifying noise: Role of randomness in generative AI</div>
    <p class="pub-authors">Gurprit Singh, Xingchang Huang, Jente Vandersanden, Cengiz Öztireli, Niloy Mitra</p>
    <p class="pub-venue">SIGGRAPH North America Courses 2025 / Eurographics Tutorial 2025</p>
    <p class="pub-links"><a href="https://diffusion-noise.mpi-inf.mpg.de/">project page</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/edge-preserving-noise-for-diffusion-models.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Edge-preserving noise for diffusion models</div>
    <p class="pub-authors">Jente Vandersanden, Sascha Holl, Xingchang Huang, Gurprit Singh</p>
    <p class="pub-venue">ICLR Workshop 2025</p>
    <p class="pub-description">Q: What would be the impact of content-aware anisotropic noise on diffusion models?</p>
    <p class="pub-links"><a href="https://edge-preserving-diffusion.mpi-inf.mpg.de/">project page</a> / <a href="https://arxiv.org/abs/2410.01540">arXiv</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/online-importance-sampling-for-stochastic-gradient-optimization.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Online importance sampling for stochastic gradient optimization</div>
    <p class="pub-authors">Corentin Salaun, Xingchang Huang, Iliyan Georgiev, Niloy Mitra, Gurprit Singh</p>
    <p class="pub-venue">ICPRAM 2025 — Best Student Paper Award</p>
    <p class="pub-description">Q: Is there an efficient way to assign importance weights to mini-batch samples in gradient estimation?</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/2025-salaun-efficient.html">project page</a> / <a href="https://arxiv.org/abs/2311.14468">arXiv</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/multiple-importance-sampling-for-stochastic-gradient-estimation.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Multiple importance sampling for stochastic gradient estimation</div>
    <p class="pub-authors">Corentin Salaun, Xingchang Huang, Iliyan Georgiev, Niloy Mitra, Gurprit Singh</p>
    <p class="pub-venue">ICPRAM 2025</p>
    <p class="pub-description">Q: What if we have multiple importance strategies for gradient estimation?</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/2025-salaun-mis.html">project page</a> / <a href="https://arxiv.org/abs/2407.15525">arXiv</a></p>
  </div>
</article>
<h3 class="publication-year">2024</h3>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/mcmc-bridging-rendering-optimization-and-generative-ai.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">MCMC: Bridging Rendering, Optimization and Generative AI</div>
    <p class="pub-authors">Gurprit Singh, Wenzel Jakob</p>
    <p class="pub-venue">SIGGRAPH Asia Courses 2024</p>
    <p class="pub-description">X: These notes are an effort to understand the role of MCMC sampling methods in rendering, optimization and generative AI.</p>
    <p class="pub-links"><a href="https://sinbag.github.io/mcmc/">project page</a> / <a href="https://arxiv.org/abs/2510.09078">arXiv</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/blue-noise-for-diffusion-models.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Blue noise for diffusion models</div>
    <p class="pub-authors">Xingchang Huang, Corentin Salaun, Cristina Vasconcelos, Christian Theobalt, Cengiz Öztireli, Gurprit Singh</p>
    <p class="pub-venue">SIGGRAPH North America 2024</p>
    <p class="pub-description">Q: How can we enhance generated samples simply from noise manipulation?</p>
    <p class="pub-links"><a href="https://xchhuang.github.io/bndm/index.html">project page</a> / <a href="https://arxiv.org/abs/2402.04930">arXiv</a></p>
  </div>
</article>
<h3 class="publication-year">2023</h3>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/joint-sampling-and-optimisation-for-inverse-rendering.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Joint sampling and optimisation for inverse rendering</div>
    <p class="pub-authors">Martin Bálint, Karol Myszkowski, Hans-Peter Seidel, Gurprit Singh</p>
    <p class="pub-venue">SIGGRAPH Asia 2023</p>
    <p class="pub-description">Q: How to reduce variance in gradient estimation during inverse rendering?</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/2023-balint-meta.html">project page</a> / <a href="https://arxiv.org/abs/2309.15676">arXiv</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/perceptual-error-optimization-for-monte-carlo-animation-rendering.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Perceptual error optimization for Monte Carlo animation rendering</div>
    <p class="pub-authors">Misa Korac*, Corentin Salaun*, Iliyan Georgiev, Pascal Grittmann, Philipp Slusallek, Karol Myszkowski, Gurprit Singh</p>
    <p class="pub-venue">SIGGRAPH Asia 2023 — conference track</p>
    <p class="pub-description">Q: How to design perceptually motivated spatio-temporal masks for Monte Carlo animation rendering?</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/2023-korac-perceptual.html">project page</a> / <a href="https://arxiv.org/abs/2310.02955">arXiv</a></p>
    <p class="pub-note">Joint first authors.</p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/patternshop-editing-point-patterns-with-image-manipulations.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Patternshop: Editing point patterns with image manipulations</div>
    <p class="pub-authors">Xingchang Huang, Tobias Ritschel, Hans-Peter Seidel, Pooran Memari, Gurprit Singh</p>
    <p class="pub-venue">SIGGRAPH North America 2023</p>
    <p class="pub-description">Q: How can we design a 2D color-space that allows editing point patterns with Photoshop?</p>
    <p class="pub-links"><a href="https://xchhuang.github.io/patternshop/">project page</a> / <a href="https://arxiv.org/abs/2308.10517">arXiv</a></p>
  </div>
</article>
<h3 class="publication-year">2022</h3>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/informatik-spektrum-scalable-multi-class-sampling.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Informatik Spektrum: Scalable multi-class sampling via filtered sliced optimal transport</div>
    <p class="pub-authors">Corentin Salaun, Iliyan Georgiev, Hans-Peter Seidel, Gurprit Singh</p>
    <p class="pub-venue">Cover image for Informatik Spektrum, October 2022</p>
    <p class="pub-links"><a href="https://link.springer.com/journal/287/volumes-and-issues/45-5">journal</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/scalable-multi-class-sampling-via-filtered-sliced-optimal-transport.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Scalable multi-class sampling via filtered sliced optimal transport</div>
    <p class="pub-authors">Corentin Salaun, Iliyan Georgiev, Hans-Peter Seidel, Gurprit Singh</p>
    <p class="pub-venue">SIGGRAPH Asia 2022 / ACM Transactions on Graphics, Volume 41, Issue 6, December 2022</p>
    <p class="pub-description">Q: How can we build a unified framework for stippling, object placement and perceptually pleasing rendering?</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/2022-salaun-multiclass.html">project page</a> / <a href="https://arxiv.org/abs/2211.04314">arXiv</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/point-pattern-synthesis-using-gabor-and-random-filters.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Point-pattern synthesis using Gabor and random filters</div>
    <p class="pub-authors">Xingchang Huang, Pooran Memari, Hans-Peter Seidel, Gurprit Singh</p>
    <p class="pub-venue">EGSR 2022 / Computer Graphics Forum, Volume 41, Issue 6, July 2022</p>
    <p class="pub-description">Q: How can we perform point pattern texture synthesis without training a network?</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/2022-huang-gabor.html">project page</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/regression-based-monte-carlo-integration.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Regression-based Monte Carlo integration</div>
    <p class="pub-authors">Corentin Salaun, Adrien Gruson, Binh-Son Hua, Toshiya Hachisuka, Gurprit Singh</p>
    <p class="pub-venue">SIGGRAPH North America 2022 / ACM Transactions on Graphics, Volume 41, Issue 4, July 2022</p>
    <p class="pub-description">Q: What happens if we use a polynomial function to average Monte Carlo estimates?</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/2022-salaun-regressionmc.html">project page</a> / <a href="https://arxiv.org/abs/2211.07422">arXiv</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/perceptual-error-optimization-for-monte-carlo-rendering.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Perceptual error optimization for Monte Carlo rendering</div>
    <p class="pub-authors">Vassillen Chizhov, Iliyan Georgiev, Karol Myszkowski, Gurprit Singh</p>
    <p class="pub-venue">ACM Transactions on Graphics, Volume 41, Issue 3, June 2022 — presented at SIGGRAPH North America 2022</p>
    <p class="pub-description">Q: How can we use a perception-based human visual system model to control the error distribution in rendering?</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/2022-chizhov-perception.html">project page</a> / <a href="https://arxiv.org/abs/2012.02344">arXiv</a></p>
  </div>
</article>
<h3 class="publication-year">2021</h3>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/informatik-spektrum-neural-light-field-3d-printing.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Informatik Spektrum: Neural Light Field 3D Printing</div>
    <p class="pub-authors">Quan Zheng, Vahid Babaei, Gordon Wetzstein, Hans-Peter Seidel, Matthias Zwicker, Gurprit Singh</p>
    <p class="pub-venue">Cover image for Informatik Spektrum, October 2021</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/publications/2021-zheng-display-image/2021-zheng-display-image.pdf">magazine</a> / <a href="https://link.springer.com/journal/287/volumes-and-issues/44-5">journal</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/neural-relightable-participating-media-rendering.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Neural Relightable Participating Media Rendering</div>
    <p class="pub-authors">Quan Zheng, Gurprit Singh, Hans-Peter Seidel</p>
    <p class="pub-venue">NeurIPS 2021</p>
    <p class="pub-links"><a href="https://arxiv.org/abs/2110.12993">arXiv</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/blue-noise-plots.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Blue Noise Plots</div>
    <p class="pub-authors">Christian van Onzenoodt, Gurprit Singh, Timo Ropinski, Tobias Ritschel</p>
    <p class="pub-venue">Eurographics 2021 / Computer Graphics Forum, Volume 40, Issue 2, May 2021</p>
    <p class="pub-links"><a href="https://arxiv.org/abs/2102.04072">arXiv</a> / <a href="https://github.com/onc/BlueNoisePlots">source code</a></p>
  </div>
</article>
<h3 class="publication-year">2020</h3>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/neural-light-field-3d-printing.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Neural Light Field 3D Printing</div>
    <p class="pub-authors">Quan Zheng, Vahid Babaei, Gordon Wetzstein, Hans-Peter Seidel, Matthias Zwicker, Gurprit Singh</p>
    <p class="pub-venue">SIGGRAPH Asia 2020 / ACM Transactions on Graphics, Volume 39, Issue 6, December 2020</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/2020-zheng-display.html">project page</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/ladybird-quasi-monte-carlo-sampling-for-deep-implicit-field-based-3d-r.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">LadyBird: Quasi-Monte Carlo Sampling for Deep Implicit Field Based 3D Reconstruction with Symmetry</div>
    <p class="pub-authors">Yifan Xu*, Tianqi Fan*, Yi Yuan, Gurprit Singh</p>
    <p class="pub-venue">ECCV 2020 — Oral</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/2020-xu-ladybird.html">project page</a></p>
    <p class="pub-note">Contributed equally.</p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/real-time-monte-carlo-denoising-with-the-neural-bilateral-grid.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Real-time Monte Carlo Denoising with the Neural Bilateral Grid</div>
    <p class="pub-authors">Xiaoxu Meng, Quan Zheng, Amitabh Varshney, Gurprit Singh, Matthias Zwicker</p>
    <p class="pub-venue">Eurographics Symposium on Rendering 2020</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/2020-xiaoxu-denoising.html">project page</a></p>
  </div>
</article>
<h3 class="publication-year">2019</h3>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/deep-point-correlation-design.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Deep Point Correlation Design</div>
    <p class="pub-authors">Thomas Leimkühler, Gurprit Singh, Karol Myszkowski, Hans-Peter Seidel, Tobias Ritschel</p>
    <p class="pub-venue">SIGGRAPH Asia 2019 / ACM Transactions on Graphics, Volume 38, Issue 6, October 2019</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/deepsampling.html">project page</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/analysis-of-sample-correlations-for-monte-carlo-rendering.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Analysis of Sample Correlations for Monte Carlo Rendering</div>
    <p class="pub-authors">Gurprit Singh, Cengiz Öztireli, Abdalla G. M. Ahmed, David Coeurjolly, Kartic Subr, Oliver Deussen, Victor Ostromoukhov, Ravi Ramamoorthi, Wojciech Jarosz</p>
    <p class="pub-venue">Computer Graphics Forum — Proceedings of Eurographics State of the Art Reports 2019</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/2019-singh-star.html">project page</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/fourier-analysis-of-correlated-monte-carlo-importance-sampling.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Fourier Analysis of Correlated Monte Carlo Importance Sampling</div>
    <p class="pub-authors">Gurprit Singh, Kartic Subr, David Coeurjolly, Victor Ostromoukhov, Wojciech Jarosz</p>
    <p class="pub-venue">Computer Graphics Forum, Volume 38, Issue 1, 2019</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/2019-singh-fourier.html">project page</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/a-perception-driven-hybrid-decomposition-for-multi-layer-accommodative.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">A Perception-driven Hybrid Decomposition for Multi-layer Accommodative Displays</div>
    <p class="pub-authors">Hyeonseung Yu, Mojtaba Bemana, Marek Wernikowski, Michał Chwesiuk, Okan Tarhan Tursun, Gurprit Singh, Karol Myszkowski, Radosław Mantiuk, Hans-Peter Seidel, Piotr Didyk</p>
    <p class="pub-venue">IEEE VR 2019</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/2019-yu-perception.html">project page</a></p>
  </div>
</article>
<h3 class="publication-year">2018</h3>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/spectral-measures-of-distortion-for-change-detection-in-dynamic-graphs.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Spectral Measures of Distortion for Change Detection in Dynamic Graphs</div>
    <p class="pub-authors">Luca Castelli Aleardi, Semih Salihoglu, Gurprit Singh, Maks Ovsjanikov</p>
    <p class="pub-venue">Complex Networks 2018 — Oral</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/2018-aleardi-spectral.html">project page</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/sampling-analysis-using-correlations-for-monte-carlo-rendering.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Sampling Analysis using Correlations for Monte Carlo Rendering</div>
    <p class="pub-authors">Cengiz Öztireli, Gurprit Singh</p>
    <p class="pub-venue">SIGGRAPH Asia Courses 2018</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/2018-oztireli-sampling.html">project page</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/end-to-end-sampling-patterns.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">End-to-end Sampling Patterns</div>
    <p class="pub-authors">Thomas Leimkühler, Gurprit Singh, Karol Myszkowski, Hans-Peter Seidel, Tobias Ritschel</p>
    <p class="pub-venue">Technical Report</p>
    <p class="pub-links"><a href="https://arxiv.org/abs/1806.06710">arXiv</a></p>
  </div>
</article>
<h3 class="publication-year">2017</h3>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/convergence-analysis-for-anisotropic-monte-carlo-sampling-spectra.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Convergence Analysis for Anisotropic Monte Carlo Sampling Spectra</div>
    <p class="pub-authors">Gurprit Singh, Wojciech Jarosz</p>
    <p class="pub-venue">SIGGRAPH 2017 / ACM Transactions on Graphics, Volume 36, Issue 4, July 2017</p>
    <p class="pub-links"><a href="https://sampling.mpi-inf.mpg.de/2017-singh-convergence.html">project page</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/variance-and-convergence-analysis-of-monte-carlo-line-and-segment-samp.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Variance and Convergence Analysis of Monte Carlo Line and Segment Samples</div>
    <p class="pub-authors">Gurprit Singh, Bailey Miller, Wojciech Jarosz</p>
    <p class="pub-venue">Computer Graphics Forum — Proceedings of EGSR, Volume 36, Issue 4, June 2017</p>
    <p class="pub-links"><a href="https://cs.dartmouth.edu/~wjarosz/publications/singh17variance.html">project page</a> / <a href="https://github.com/sinbag/ao-line-segment-sampling">source code</a></p>
  </div>
</article>
<h3 class="publication-year">2016</h3>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/monte-carlo-convergence-analysis-for-anisotropic-sampling-power-spectr.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Monte Carlo Convergence Analysis for Anisotropic Sampling Power Spectra</div>
    <p class="pub-authors">Gurprit Singh, Wojciech Jarosz</p>
    <p class="pub-venue">Technical Report</p>
    <p class="pub-links"><a href="https://cs.dartmouth.edu/~wjarosz/publications/singh16monte.html">technical report</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/fourier-analysis-of-numerical-integration-in-monte-carlo-rendering-the.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Fourier Analysis of Numerical Integration in Monte Carlo Rendering: Theory and Practice</div>
    <p class="pub-authors">Kartic Subr, Gurprit Singh, Wojciech Jarosz</p>
    <p class="pub-venue">SIGGRAPH Courses 2016</p>
    <p class="pub-links"><a href="https://cs.dartmouth.edu/~wjarosz/publications/subr16fourier.html">project page</a> / <a href="https://dl.acm.org/doi/10.1145/2897826.2927356">ACM</a> / <a href="https://github.com/sinbag/EmpiricalErrorAnalysis">source code</a></p>
  </div>
</article>
<h3 class="publication-year">2015</h3>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/variance-and-sampling-analysis-spherical-domain.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Variance and Sampling Analysis for Monte Carlo Integration in the Spherical Domain</div>
    <p class="pub-authors">Gurprit Singh</p>
    <p class="pub-venue">Ph.D. Dissertation, Université Lyon 1, France, September 2015</p>
    <p class="pub-links"><a href="https://hal.archives-ouvertes.fr/tel-01217082">HAL</a></p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/variance-analysis-for-monte-carlo-integration.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Variance Analysis for Monte Carlo Integration</div>
    <p class="pub-authors">Adrien Pilleboue*, Gurprit Singh*, David Coeurjolly, Michael Kazhdan, Victor Ostromoukhov</p>
    <p class="pub-venue">SIGGRAPH 2015 / ACM Transactions on Graphics, Volume 34, Issue 4, 2015</p>
    <p class="pub-links"><a href="https://projet.liris.cnrs.fr/variance/">project page</a> / <a href="https://dl.acm.org/doi/10.1145/2766930">ACM</a> / <a href="https://github.com/stk-team/stk">source code</a></p>
    <p class="pub-note">Joint first authors.</p>
  </div>
</article>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/variance-analysis-representation-theoretic-perspective.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Variance Analysis for Monte Carlo Integration: A Representation-Theoretic Perspective</div>
    <p class="pub-authors">Michael Kazhdan, Gurprit Singh, Adrien Pilleboue, David Coeurjolly, Victor Ostromoukhov</p>
    <p class="pub-venue">Technical Report</p>
    <p class="pub-links"><a href="https://arxiv.org/abs/1506.00021">arXiv</a></p>
  </div>
</article>
<h3 class="publication-year">2014</h3>
<article class="pub-card">
  <div class="pub-thumb" style="background-image: url('{{ '/assets/images/publications/fast-tile-based-adaptive-sampling-with-user-specified-fourier-spectra.jpg' | relative_url }}');" aria-hidden="true"></div>
  <div class="pub-body">
    <div class="pub-title">Fast Tile-Based Adaptive Sampling with User-Specified Fourier Spectra</div>
    <p class="pub-authors">Florent Wachtel, Adrien Pilleboue, David Coeurjolly, Katherine Breeden, Gurprit Singh, Gaël Cathelin, Fernando de Goes, Mathieu Desbrun, Victor Ostromoukhov</p>
    <p class="pub-venue">SIGGRAPH 2014 / ACM Transactions on Graphics, Volume 33, Issue 4, 2014</p>
    <p class="pub-links"><a href="https://projet.liris.cnrs.fr/polyhex/">project page</a> / <a href="https://dl.acm.org/doi/10.1145/2601097.2601107">ACM</a></p>
  </div>
</article>
</section>

<section id="contact" class="about-section">
<h2>Contact</h2>
<p>The best way to reach me is by <a href="mailto:gurpritsbagga@gmail.com">email</a>. You can also find me on <a href="https://github.com/sinbag">GitHub</a> and <a href="https://www.linkedin.com/in/sinbag/">LinkedIn</a>.</p>
</section>

</div>
