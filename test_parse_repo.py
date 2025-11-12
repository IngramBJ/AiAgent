"""
Test cases for parse_repo function to ensure it handles various GitHub URL formats.
"""
import pytest
import re


def parse_repo(url: str):
    """Parse GitHub URL to extract owner, repo, and subdir."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)(/.*)?", url.strip())
    if not m: raise ValueError("repo_url 需形如 https://github.com/owner/repo")
    owner, repo, extra = m.group(1), m.group(2), m.group(3) or ""
    
    # Strip .git suffix from repo if present
    if repo.endswith('.git'):
        repo = repo[:-4]
    
    # Handle /tree/<branch>/<path> URLs
    extra = extra.strip("/")
    if extra:
        parts = extra.split("/")
        # If URL contains /tree/, extract the path after the branch
        if len(parts) >= 2 and parts[0] == "tree":
            # parts[1] is the branch name, everything after is the subdir
            if len(parts) > 2:
                subdir = "/".join(parts[2:])
            else:
                subdir = ""
        else:
            # Legacy format: /path/to/dir without /tree/
            subdir = extra.split("/", 2)[-1] if extra else ""
    else:
        subdir = ""
    
    return owner, repo, subdir


def test_parse_repo_simple():
    """Test basic GitHub URL parsing."""
    owner, repo, subdir = parse_repo("https://github.com/owner/repo")
    assert owner == "owner"
    assert repo == "repo"
    assert subdir == ""


def test_parse_repo_with_http():
    """Test parsing with http:// protocol."""
    owner, repo, subdir = parse_repo("http://github.com/owner/repo")
    assert owner == "owner"
    assert repo == "repo"
    assert subdir == ""


def test_parse_repo_with_git_suffix():
    """Test parsing GitHub URL with .git suffix."""
    owner, repo, subdir = parse_repo("https://github.com/owner/repo.git")
    assert owner == "owner"
    assert repo == "repo"
    assert subdir == ""


def test_parse_repo_with_git_suffix_and_trailing_slash():
    """Test parsing GitHub URL with .git suffix and trailing slash."""
    owner, repo, subdir = parse_repo("https://github.com/owner/repo.git/")
    assert owner == "owner"
    assert repo == "repo"
    assert subdir == ""


def test_parse_repo_with_tree_branch():
    """Test parsing GitHub URL with /tree/<branch>."""
    owner, repo, subdir = parse_repo("https://github.com/owner/repo/tree/main")
    assert owner == "owner"
    assert repo == "repo"
    assert subdir == ""


def test_parse_repo_with_tree_branch_and_subdir():
    """Test parsing GitHub URL with /tree/<branch>/<path>."""
    owner, repo, subdir = parse_repo("https://github.com/owner/repo/tree/main/src")
    assert owner == "owner"
    assert repo == "repo"
    assert subdir == "src"


def test_parse_repo_with_tree_branch_and_nested_subdir():
    """Test parsing GitHub URL with /tree/<branch>/<nested/path>."""
    owner, repo, subdir = parse_repo("https://github.com/owner/repo/tree/main/src/core/utils")
    assert owner == "owner"
    assert repo == "repo"
    assert subdir == "src/core/utils"


def test_parse_repo_with_tree_branch_name_with_slash():
    """Test parsing GitHub URL with branch name containing slash.
    
    Note: Since we can't distinguish branch names with slashes from paths without
    querying GitHub's API, we treat everything after /tree/ as potential path components.
    This is a known limitation when branch names contain slashes.
    """
    owner, repo, subdir = parse_repo("https://github.com/owner/repo/tree/feature/new-feature")
    assert owner == "owner"
    assert repo == "repo"
    # When URL is /tree/feature/new-feature, we treat "feature" as branch and "new-feature" as path
    # This is a limitation - branch names with slashes need special handling
    assert subdir == "new-feature"


def test_parse_repo_with_tree_branch_name_with_slash_and_path():
    """Test parsing GitHub URL with branch name containing slash and additional path.
    
    Note: Since we can't distinguish branch names with slashes from paths without
    querying GitHub's API, this is a known limitation when branch names contain slashes.
    """
    owner, repo, subdir = parse_repo("https://github.com/owner/repo/tree/feature/new-feature/src/main")
    assert owner == "owner"
    assert repo == "repo"
    # When URL is /tree/feature/new-feature/src/main, we treat "feature" as branch
    # and "new-feature/src/main" as path. This is a limitation.
    assert subdir == "new-feature/src/main"


def test_parse_repo_with_subdir_no_tree():
    """Test parsing GitHub URL with subdir but no /tree/ (legacy format)."""
    owner, repo, subdir = parse_repo("https://github.com/owner/repo/subdir")
    assert owner == "owner"
    assert repo == "repo"
    assert subdir == "subdir"


def test_parse_repo_invalid_url():
    """Test that invalid URLs raise ValueError."""
    with pytest.raises(ValueError):
        parse_repo("not-a-github-url")


def test_parse_repo_with_whitespace():
    """Test that URLs with whitespace are handled correctly."""
    owner, repo, subdir = parse_repo("  https://github.com/owner/repo  ")
    assert owner == "owner"
    assert repo == "repo"
    assert subdir == ""


def test_parse_repo_with_git_suffix_and_tree():
    """Test parsing GitHub URL with .git suffix and /tree/."""
    owner, repo, subdir = parse_repo("https://github.com/owner/repo.git/tree/main/src")
    assert owner == "owner"
    assert repo == "repo"
    assert subdir == "src"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
