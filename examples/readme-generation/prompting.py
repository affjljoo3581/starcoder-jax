from __future__ import annotations

import glob
import os
import random

from transformers import PreTrainedTokenizerBase

DIALOGUE_SEPARATOR = "&&&&&&"

IMAGE_EXTENSIONS = (
    "ase,art,bmp,blp,cd5,cit,cpt,cr2,cut,dds,dib,djvu,egt,exif,gif,gpl,grf,icns,ico,"
    "iff,jng,jpeg,jpg,jfif,jp2,jps,lbm,max,miff,mng,msp,nef,nitf,ota,pbm,pc1,pc2,pc3,"
    "pcf,pcx,pdn,pgm,PI1,PI2,PI3,pict,pct,pnm,pns,ppm,psb,psd,pdd,psp,px,pxm,pxr,qfx,"
    "raw,rle,sct,sgi,rgb,int,bw,tga,tiff,tif,vtf,xbm,xcf,xpm,3dv,amf,ai,awg,cgm,cdr,"
    "cmx,dxf,e2d,egt,eps,fs,gbr,odg,svg,stl,vrml,x3d,sxd,v2d,vnd,wmf,emf,art,xar,png,"
    "webp,jxr,hdp,wdp,cur,ecw,iff,lbm,liff,nrrd,pam,pcx,pgf,sgi,rgb,rgba,bw,int,inta,"
    "sid,ras,sun,tga,heic,heif"
)
IMAGE_EXTENSIONS = IMAGE_EXTENSIONS.lower().split(",")


def get_filenames_from_repo(repo_dir: str) -> tuple[list[str], list[str]]:
    filenames = glob.glob(os.path.join(repo_dir, "**/*.*"), recursive=True)
    filenames = [filename for filename in filenames if os.path.isfile(filename)]
    filenames = [filename[len(repo_dir.rstrip("/")) + 1 :] for filename in filenames]

    source_codes, image_files = [], []
    for filename in filenames:
        if os.path.splitext(filename)[-1][1:].lower() in IMAGE_EXTENSIONS:
            image_files.append(filename)
        else:
            source_codes.append(filename)

    return source_codes, image_files


def sample_random_files_for_prompt(
    filenames: list[str], tokenizer: PreTrainedTokenizerBase, max_length: int = 256
) -> list[str]:
    random.shuffle(filenames)
    prompt = "- "
    prompt += "\n- ".join(filenames)

    prompt = tokenizer.decode(tokenizer.encode(prompt)[:max_length])
    prompt = prompt.splitlines()
    return "\n".join(prompt[:-1] if len(prompt) > 1 else prompt)


def read_source_code_for_prompt(
    repo_dir: str,
    filename: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> str:
    try:
        with open(os.path.join(repo_dir, filename)) as fp:
            content = fp.read()
    except UnicodeDecodeError:
        return ""

    preprompt = f"{DIALOGUE_SEPARATOR}\n" f"$ head -n @@@ {filename}\n" f"{content}\n"
    preprompt = tokenizer.decode(tokenizer.encode(preprompt)[:max_length])
    preprompt = preprompt.splitlines()
    preprompt = "\n".join(preprompt[2:-1] if len(preprompt) > 3 else preprompt[2:])

    prompt = (
        f"{DIALOGUE_SEPARATOR}\n"
        f"$ head -n {len(preprompt.splitlines())} {filename}\n"
        f"{preprompt}\n"
    )
    return prompt


def create_input_prompt(
    repo_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    git_url: str | None = None,
    max_length: int = 7168,
) -> str:
    source_codes, image_files = get_filenames_from_repo(repo_dir)
    source_codes_preview = sample_random_files_for_prompt(source_codes, tokenizer, 512)
    image_files_preview = sample_random_files_for_prompt(image_files, tokenizer, 512)

    if git_url is None:
        git_url = os.popen(f"cd {repo_dir}; git config --get remote.origin.url").read()

    prompt = (
        f"Below are a series of linux terminal outputs on a repository directory. "
        f"The dialoges contain part of source codes in the repository. "
        f"Each file must be highly related to other source codes and documents should "
        f"contain proper information about the project. "
        f"The documents should not have false or misleading contents.\n"
        f"\n"
        f"The repository contains {len(source_codes)} source codes. "
        f"The below is a list of some source code paths:\n"
        f"{source_codes_preview}\n"
        f"\n"
        f"The repository contains {len(image_files)} image files. "
        f"The below is a list of some image paths:\n"
        f"{image_files_preview}\n"
        f"{DIALOGUE_SEPARATOR}\n"
        f"$ git config --get remote.origin.url\n"
        f"{git_url}\n"
    )

    postfix = (
        f"{DIALOGUE_SEPARATOR}\n"
        f"$ ls -lh README.md\n"
        f"-rw-r--r-- 1 user user 3.5K Jun  9 14:55 README.md\n"
        f"{DIALOGUE_SEPARATOR}\n"
        f"$ cat README.md\n"
    )
    postfix_length = len(tokenizer.tokenize(postfix))

    if "README.md" in source_codes:
        source_codes.remove("README.md")
    random.shuffle(source_codes)

    while (left := max_length - len(tokenizer.tokenize(prompt)) - postfix_length) > 128:
        if not source_codes:
            break
        prompt += read_source_code_for_prompt(
            repo_dir, source_codes.pop(0), tokenizer, max_length=min(512, left)
        )
    return prompt + postfix
