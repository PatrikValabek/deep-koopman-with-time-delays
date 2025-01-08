# koopman_dmd_vs_derek

## Paper TODOS

* see multiple "%%MKL" in tex source
* change the paper title, avoid abbreviations which nobody understands
* include "young researcher grant"
* ChatGPT feeling is strong, e.g. word "Leveraging"
* Abstract is too general, try to rewrite to be more specific to the theory and results from the paper. Basically, good abstract summarizes key points of the paper and provide some sort of "marketing" value
* abbreviations in keywords make no sense
* "traction in recent years~\cite{Mezic2004101, Mezić2005}" -> traction in recent years, ale cialnky su 10 rokov stare
* "introduced by Schmid~\cite{schmid2010dynamic}" -> mena autorov nepouzivat, ved je citacia
* Introduction: a term "dictionary" is often used, I'm missing some sort of "definition". Keep in mind, the audience of the conference, and such a term might not be well-known
* Rename "Methodology" to content-specific chapter name
* we are missing a formal, mathematical definitions of the Koopman based model and its use in simulation
* Aviod terms like "unnecessary huge matrix", what is a huge matrix? 
* Rename "Results" to something like "Case Study", "Numerical Examples" etc. Avoid "generaly speaking section titles"
* Learn to use `siunitx` package... "\text{m}^3\text{s}^{-1}"
* Table I: unusable, see comments in tex
* General comment: the paper is missing clear mathematical definitions of key vectors, matrices in equations.
  
## Future recommendations
* general suggested labeling in LaTeX: `env_identificator:chapter_label:section_label:actual_label:sub_actual_label`
  * in conference papers: `env_identificator:section_label:actual_label`
  * in subequations/align: `eq:section_label:actual_label` in main environment and then `eq:section_label:actual_label:label_of_individual_eq`

## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.uiam.sk/valabek/koopman_dmd_vs_derek.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.uiam.sk/valabek/koopman_dmd_vs_derek/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***
