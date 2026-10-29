"""Regression tests: sudo privilege-flag rule reads only SUDO'S OWN options.

t_4209213f / t_cc5a5d28 / t_a652606d: any sudo'd command containing the
substring ``--storage`` (``sudo vzdump 205 --storage backup-nfs``, ``pvesm
... --storage``) was refused by the terminal safety classifier as a "sudo
shell/list privilege flag". Root cause: two DANGEROUS_PATTERNS rules scanned
the WHOLE command segment with a lazy ``[^;|&\\n]*?``, so a flag belonging to
the WRAPPED PROGRAM (``--storage``) was attributed to sudo itself (it matched
the ``--st[a-z]*`` alternative intended for ``--stdin``). sudo stops parsing
its OWN options at the first non-option word -- that word is the program it
runs -- so a flag after it belongs to the program, not to sudo.

In a headless ``-q`` kanban worker a false positive here is a hard stop with
no human present to approve it (tech hit this twice on pve1 backup work and
had to resort to writing the command to a script file, which hides the
command from review).

Fix: the flag run is now bounded to sudo's own recognised option tokens
(``_SUDO_OWN_FLAG_RUN``), which stop at the first word that isn't one of
sudo's own options.

A second, related gap closed in the same change: the rule's own description
promises "stdin/askpass/shell/list" coverage, but the alternation only ever
wired ``-s`` (shell, short) and ``-a``/``--a*`` (askpass). ``--shell`` (long
form) and ``-l``/``--list`` (sudo's actual list-privileges option) were never
matched at all, so ``sudo --shell`` and ``sudo -l`` sailed through ungated.

Every test here asserts BOTH directions -- the negative control (the wrapped
program's flag) is the half that matters; a "fix" that merely stopped
matching sudo altogether would pass a one-sided test while deleting real
coverage.
"""

import pytest

from tools.approval import detect_dangerous_command

_SUDO_KEY = "sudo with privilege flag (stdin/askpass/shell/list)"
_SUDO_COMBINED_KEY = "sudo with combined-flag privilege escalation"


def _fires(command, key):
    """Whether *command* is gated specifically by the rule keyed *key*.

    Checks the rule, not just the verdict: several of these commands could be
    dangerous for OTHER (correct) reasons, and a test that only looked at the
    boolean could not tell a fixed rule from a still-broken one.
    """
    is_dangerous, pattern_key, _ = detect_dangerous_command(command)
    return is_dangerous and pattern_key == key


def _fires_either_sudo_rule(command):
    is_dangerous, pattern_key, _ = detect_dangerous_command(command)
    return is_dangerous and pattern_key in (_SUDO_KEY, _SUDO_COMBINED_KEY)


class TestSudoPrivilegeFlagsBelongToSudo:
    """Only sudo's OWN options are privilege flags; the wrapped program's are not."""

    # -- the card's exact repro ----------------------------------------------

    @pytest.mark.parametrize(
        "command",
        [
            "sudo vzdump 205 --storage backup-nfs",
            "sudo pvesm free backup:dump/x.tar --storage backup",
        ],
    )
    def test_card_repro_storage_flag_not_read_as_sudo_stdin(self, command):
        """The reported bug: ``--storage`` matched the ``--stdin`` alternative."""
        assert not _fires_either_sudo_rule(command), (
            f"wrapped program's --storage flag read as a sudo privilege flag: {command!r}"
        )

    # -- the wrapped program's flags must NOT be gated -----------------------

    @pytest.mark.parametrize(
        "command",
        [
            "sudo vzdump 205 --storage backup-nfs",
            "sudo pvesm free backup:dump/x.tar --storage backup",
            "sudo ls -a /root",
            "sudo grep -a pattern /var/log/syslog",
            "sudo systemctl --state=failed list-units",
            "sudo tar -a -cf archive.tar /etc",
            "sudo apt-get --assume-yes install foo",
            "sudo -u postgres psql --single-transaction -f dump.sql",
            # The two flags this fix newly recognises as sudo's own (-l / --shell)
            # must still not fire off a WRAPPED PROGRAM's --list / --shell-alike flag.
            "sudo tar --list -f archive.tar",
            "sudo systemctl --show-types list-units",
        ],
    )
    def test_wrapped_program_flags_not_attributed_to_sudo(self, command):
        assert not _fires_either_sudo_rule(command), (
            f"wrapped program's flag attributed to sudo: {command!r}"
        )

    # -- sudo's own privilege flags must STILL be gated (true positives) ----

    @pytest.mark.parametrize(
        "command",
        [
            "sudo -S whoami",
            "sudo -s",
            "sudo -a whoami",
            "sudo --stdin whoami",
            "sudo --askpass whoami",
            "sudo -u root -S whoami",
            "sudo -S docker ps",
        ],
    )
    def test_sudo_own_privilege_flags_still_gated(self, command):
        assert _fires(command, _SUDO_KEY), (
            f"sudo privilege flag no longer gated: {command!r}"
        )

    @pytest.mark.parametrize(
        "command",
        [
            "sudo --shell",
            "sudo --shell whoami",
            "sudo -l",
            "sudo -l -U root",
            "sudo --list",
        ],
    )
    def test_sudo_shell_and_list_now_gated(self, command):
        """The card's acceptance criteria: ``sudo -s``, ``sudo --shell``, ``sudo -l``.

        ``-s`` already worked; ``--shell`` (long form) and ``-l``/``--list``
        (sudo's actual privilege-list option) were gaps in the SAME rule --
        its description already claimed "shell/list" coverage.
        """
        assert _fires(command, _SUDO_KEY), (
            f"sudo shell/list privilege flag no longer gated: {command!r}"
        )

    @pytest.mark.parametrize(
        "command",
        [
            "sudo --st whoami",
            "sudo --std whoami",
            "sudo --stdi whoami",
            "sudo --askpas whoami",
            "sudo --a whoami",
            "sudo --she whoami",
            "sudo --shel whoami",
            "sudo --lis",
            "sudo --li",
        ],
    )
    def test_sudo_long_option_prefixes_still_gated(self, command):
        """sudo resolves unambiguous long-option prefixes via getopt_long.

        Verified against sudo 1.9.15p5 (``sudo --help``): ``--stdin``,
        ``--shell`` and ``--list`` are the only long options starting with
        ``st``/``sh``/``li`` respectively, and ``--askpass`` the only one
        starting with ``a``. So the prefix classes must NOT be narrowed to
        the full spellings -- ``sudo --std cmd`` really does read the
        password from stdin, ``sudo --she`` really does run a shell.
        """
        assert _fires(command, _SUDO_KEY), (
            f"sudo long-option prefix no longer gated: {command!r}"
        )

    @pytest.mark.parametrize(
        "command",
        [
            "sudo -p 'pw: ' -S whoami",
            'sudo -p "enter pw: " --stdin whoami',
            "sudo --prompt='pw: ' --stdin id",
            "sudo --prompt 'pw: ' -S id",
            "sudo -u root -l",
        ],
    )
    def test_privilege_flag_after_a_quoted_option_argument_still_gated(self, command):
        """A sudo option argument may be QUOTED and contain spaces.

        ``sudo -p 'pw: ' -S whoami`` still pipes the password via stdin. Scoping
        the flag run to sudo's own options must therefore treat a quoted
        ``'pw: '`` as ONE token -- stopping at the space inside the quotes would
        end the run early and drop the ``-S`` that follows.
        """
        assert _fires_either_sudo_rule(command), (
            f"privilege flag after a quoted option argument no longer gated: {command!r}"
        )

    def test_plain_sudo_not_gated(self):
        """Pre-existing negative must not regress."""
        assert detect_dangerous_command(
            "sudo apt-get install -y nginx"
        ) == (False, None, None)

    def test_mention_in_a_string_literal_not_gated_via_this_file_scope(self):
        """Out of scope for this fix, recorded so a future change doesn't assume it:

        ``sudo`` inside an unrelated quoted string, followed by an unrelated
        flag later in the SAME unquoted segment, is a DIFFERENT (also real)
        false-positive shape this fix does not address -- quote-masking the
        sudo rule itself was judged out of scope here because sudo's own rule
        has no command-name token to anchor the way docker/mkfs do. Left as a
        known gap, not silently "fixed" by accident.
        """
        # Documented, not asserted -- this is a known follow-up, not a regression gate.
        pass
