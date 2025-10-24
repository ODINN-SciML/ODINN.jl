# How to contribute

We welcome all types of contributions to the ODINN ecosystem! Most of the code, documentation, and features of ODINN are constantly under development, so your feedback can be very helpful! If you are interested in contributing, there are many ways in which you can help:

- **Report bugs in the code.** You can report problems with the code by opening issues under the [issues](https://github.com/ODINN-SciML/ODINN.jl/issues) tab in the ODINN repository. Please explain the problem you encounter and try to give a complete description of it so we can follow up on that.
- **Request new features and documentation.** If there is an important topic or example that you feel falls under the scope of this project and you would like us to include it, please request it! We are looking for new insights into what the community wants to learn.
- **Contribute to the source code.** We welcome pull requests (PRs) to any  of the libraries in the ODINN ecosystem. In order to contribute, please make a fork of the repository you would like to contribute to, and then submit a PR to:
  - the `dev` branch in [ODINN.jl](https://github.com/ODINN-SciML/ODINN.jl/) (`main` is only updated from time to time when enough meaningful commits are available to perform a new release);
  - the `main` branch in [Sleipnir.jl](https://github.com/ODINN-SciML/Sleipnir.jl), [Muninn.jl](https://github.com/ODINN-SciML/Muninn.jl) and [Huginn.jl](https://github.com/ODINN-SciML/Huginn.jl/).

We will review your PR it and provide feedback. If you are looking for ideas of how to contribute with code, you can check the [opened issues](https://github.com/ODINN-SciML/ODINN.jl/issues) in our repositories.

!!! tip
    If you need help navigating the world of PRs and contributing in GitHub, we encourage you to take a look at the [tutorial](https://docs.oggm.org/en/stable/contributing.html) put together by our OGGM friends.

## Contributing to the documentation

Here we show the basics around building the docs locally and making contributions.

!!! warning "Multiprocessing in the documentation"

    In order to use multiprocessing in the documentation, we set up a specific number of workers in the Julia session in the [`documentation.yml`](https://github.com/ODINN-SciML/ODINN.jl/blob/main/.github/workflows/documentation.yml) file. It is imperative that the number of workers set there matches the ones set in the Julia code run in the documentation. By default, we have set them to `-p 3` in `documentation.yml`, meaning that 3 workers will be added on top of the head one. This will match the default number of workers in `SimulationParameters`, but if you manually specify them, make sure to set them to 4 (the number of parameters in ODINN DOES include the head worker). This is often a source of confusion, so refrain from playing with the number of workers in the documentation. 

### Running the documentation in local

This section contains the instructions to run the documentation locally.

To generate the documentation on your local computer, in the `docs/` folder run:
```julia
include("make.jl")
```

Then in another REPL, in the `docs/` folder, activate the docs environment and run the server:
```julia
using Pkg
Pkg.activate()
using LiveServer
serve()
```

This will print a localhost URL that you can open in your browser to visualize the documentation.

!!! note "What to do when it freezes?
    If the building of the documentation freezes, there can be several reasons that cause this. First try to run `include("testdocs.jl")` which will run the tutorial examples. If there is an error during the execution, this will be easier to spot it as [Literate.jl](https://github.com/fredrikekre/Literate.jl) does not always report the error. If after making sure that the code runs smoothly this still freezes, inspect the generated `.md` files (see the list of files at the beginning of `make.jl`) and check that the markdown file was generated properly (code in `@example` sections).


## Code of conduct

We want everyone in the ODINN community to feel safe and respected.
Our code of conduct outlines the standards we expect from all contributors and maintainers—please take a moment to read it before getting involved.
You can find the code of conduct in our [GitHub repository](https://github.com/ODINN-SciML/ODINN.jl) under `docs/src/code_of_conduct.md`:

```
# Contributor Covenant 3.0 Code of Conduct

## Our Pledge

We pledge to make our community welcoming, safe, and equitable for all.

We are committed to fostering an environment that respects and promotes the dignity, rights, and contributions of all individuals, regardless of characteristics including race, ethnicity, caste, color, age, physical characteristics, neurodiversity, disability, sex or gender, gender identity or expression, sexual orientation, language, philosophy or religion, national or social origin, socio-economic position, level of education, or other status. The same privileges of participation are extended to everyone who participates in good faith and in accordance with this Covenant.

## Encouraged Behaviors

While acknowledging differences in social norms, we all strive to meet our community's expectations for positive behavior. We also understand that our words and actions may be interpreted differently than we intend based on culture, background, or native language.

With these considerations in mind, we agree to behave mindfully toward each other and act in ways that center our shared values, including:

1. Respecting the **purpose of our community**, our activities, and our ways of gathering.
2. Engaging **kindly and honestly** with others.
3. Respecting **different viewpoints** and experiences.
4. **Taking responsibility** for our actions and contributions.
5. Gracefully giving and accepting **constructive feedback**.
6. Committing to **repairing harm** when it occurs.
7. Behaving in other ways that promote and sustain the **well-being of our community**.


## Restricted Behaviors

We agree to restrict the following behaviors in our community. Instances, threats, and promotion of these behaviors are violations of this Code of Conduct.

1. **Harassment.** Violating explicitly expressed boundaries or engaging in unnecessary personal attention after any clear request to stop.
2. **Character attacks.** Making insulting, demeaning, or pejorative comments directed at a community member or group of people.
3. **Stereotyping or discrimination.** Characterizing anyone’s personality or behavior on the basis of immutable identities or traits.
4. **Sexualization.** Behaving in a way that would generally be considered inappropriately intimate in the context or purpose of the community.
5. **Violating confidentiality**. Sharing or acting on someone's personal or private information without their permission.
6. **Endangerment.** Causing, encouraging, or threatening violence or other harm toward any person or group.
7. Behaving in other ways that **threaten the well-being** of our community.

### Other Restrictions

1. **Misleading identity.** Impersonating someone else for any reason, or pretending to be someone else to evade enforcement actions.
2. **Failing to credit sources.** Not properly crediting the sources of content you contribute.
3. **Promotional materials**. Sharing marketing or other commercial content in a way that is outside the norms of the community.
4. **Irresponsible communication.** Failing to responsibly present content which includes, links or describes any other restricted behaviors.


## Reporting an Issue

Tensions can occur between community members even when they are trying their best to collaborate. Not every conflict represents a code of conduct violation, and this Code of Conduct reinforces encouraged behaviors and norms that can help avoid conflicts and minimize harm.

When an incident does occur, it is important to report it promptly. To report a possible violation, **please reach out to any member of the developer team.**

Community Moderators take reports of violations seriously and will make every effort to respond in a timely manner. They will investigate all reports of code of conduct violations, reviewing messages, logs, and recordings, or interviewing witnesses and other participants. Community Moderators will keep investigation and enforcement actions as transparent as possible while prioritizing safety and confidentiality. In order to honor these values, enforcement actions are carried out in private with the involved parties, but communicating to the whole community may be part of a mutually agreed upon resolution.


## Addressing and Repairing Harm

****

If an investigation by the Community Moderators finds that this Code of Conduct has been violated, the following enforcement ladder may be used to determine how best to repair harm, based on the incident's impact on the individuals involved and the community as a whole. Depending on the severity of a violation, lower rungs on the ladder may be skipped.

1) Warning
   1) Event: A violation involving a single incident or series of incidents.
   2) Consequence: A private, written warning from the Community Moderators.
   3) Repair: Examples of repair include a private written apology, acknowledgement of responsibility, and seeking clarification on expectations.
2) Temporarily Limited Activities
   1) Event: A repeated incidence of a violation that previously resulted in a warning, or the first incidence of a more serious violation.
   2) Consequence: A private, written warning with a time-limited cooldown period designed to underscore the seriousness of the situation and give the community members involved time to process the incident. The cooldown period may be limited to particular communication channels or interactions with particular community members.
   3) Repair: Examples of repair may include making an apology, using the cooldown period to reflect on actions and impact, and being thoughtful about re-entering community spaces after the period is over.
3) Temporary Suspension
   1) Event: A pattern of repeated violation which the Community Moderators have tried to address with warnings, or a single serious violation.
   2) Consequence: A private written warning with conditions for return from suspension. In general, temporary suspensions give the person being suspended time to reflect upon their behavior and possible corrective actions.
   3) Repair: Examples of repair include respecting the spirit of the suspension, meeting the specified conditions for return, and being thoughtful about how to reintegrate with the community when the suspension is lifted.
4) Permanent Ban
   1) Event: A pattern of repeated code of conduct violations that other steps on the ladder have failed to resolve, or a violation so serious that the Community Moderators determine there is no way to keep the community safe with this person as a member.
   2) Consequence: Access to all community spaces, tools, and communication channels is removed. In general, permanent bans should be rarely used, should have strong reasoning behind them, and should only be resorted to if working through other remedies has failed to change the behavior.
   3) Repair: There is no possible repair in cases of this severity.

This enforcement ladder is intended as a guideline. It does not limit the ability of Community Managers to use their discretion and judgment, in keeping with the best interests of our community.


## Scope

This Code of Conduct applies within all community spaces, and also applies when an individual is officially representing the community in public or other spaces. Examples of representing our community include using an official email address, posting via an official social media account, or acting as an appointed representative at an online or offline event.


## Attribution

This Code of Conduct is adapted from the Contributor Covenant, version 3.0, permanently available at [https://www.contributor-covenant.org/version/3/0/](https://www.contributor-covenant.org/version/3/0/).

Contributor Covenant is stewarded by the Organization for Ethical Source and licensed under CC BY-SA 4.0. To view a copy of this license, visit [https://creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/)

For answers to common questions about Contributor Covenant, see the FAQ at [https://www.contributor-covenant.org/faq](https://www.contributor-covenant.org/faq). Translations are provided at [https://www.contributor-covenant.org/translations](https://www.contributor-covenant.org/translations). Additional enforcement and community guideline resources can be found at [https://www.contributor-covenant.org/resources](https://www.contributor-covenant.org/resources). The enforcement ladder was inspired by the work of [Mozilla’s code of conduct team](https://github.com/mozilla/inclusion).

```
