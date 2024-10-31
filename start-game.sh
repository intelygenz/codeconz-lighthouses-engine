#!/usr/bin/env bash
###############################################################################
#               ::: CodeconZ 2024 ::: LighthouseS AI Contest :::              #
###############################################################################
#
# Usage:
#   ./start-game.sh  [-r] -f <configfile>   run a real game round
#   ./start-game.sh  [-r] -d                dry-run (simulation)
#
# Options:
#
#   -r    force rebuild of the game server image (optional)
#   -f    specify game-config file REQUIRED for a real game round
#   -d    dry-run: a 100% fake playground. It builds the
#         server + 6 dummy bots, picks a random map and plays 1 round
#
#   The game-config file format must be:
#
#     # bots array with all round participants, these are docker pull URIs,
#     # these images MUST be public. 'latest' tag will always be used for pulling, ge:
#     bots=('ghcr.io/john/bot-foo' 'docker.io/jane/bot-bar' ... 'quay.io/dave/bot-baz')
#
#     # map file. must exist into ./maps/ folder, ge:
#     map=square.txt
#
###############################################################################

# Configuration
GAME_SERVER_NAME='game'
GAME_SERVER_PORT='50051'
DOCKER_NETWORK='game_net'
BOT_DEFAULT_PORT='3001'

###############################################################################

REPO_DIR="$(git rev-parse --show-toplevel)"
MAPS_DIR="${REPO_DIR}/maps"
LOG_DIR="${REPO_DIR}/log"
OUTPUT_DIR="${REPO_DIR}/output"
GAME_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
DOCKERFILE_GAME="${REPO_DIR}/docker/Dockerfile.game"
DOCKER_COMPOSE_FILE="game-${GAME_TIMESTAMP}.yaml"
LOG_FILE="${LOG_DIR}/game-${GAME_TIMESTAMP}.log"

# yay colors
BLUE="\033[1;94m"
BOLD="\033[1;37m"
CLEAR="\033[0m"
CYAN="\033[1;96m"
GREEN="\033[1;92m"
MAGENTA="\033[1;95m"
RED="\033[1;31m"
YELLOW="\033[1;93m"

set -a

function _help() {
	echo -e "
Usage:
  ./$(basename ${0}) [-r] -f <configfile>   ${MAGENTA}run a real game round${CLEAR}
  ./$(basename ${0}) [-r] -d                ${CYAN}dry-run (simulation)${CLEAR}

Options:

  -r    force rebuild of the game server image (optional)
  -f    specify game-config file ${MAGENTA}REQUIRED for a real game round${CLEAR}
  -d    ${CYAN}dry-run${CLEAR}: a 100% fake playground. It builds the
        server + 6 dummy bots, picks a random map and plays 1 round

  The game-config file format must be:

    # bots array with all round participants, these are docker pull URIs,
    # these images MUST be public. 'latest' tag will always be used for pulling, ge:
    bots=(${YELLOW}'ghcr.io/john/bot-foo' 'docker.io/jane/bot-bar' ... 'quay.io/dave/bot-baz'${CLEAR})

    # map file. must exist into ./maps/ folder, ge:
    map=${YELLOW}square.txt${CLEAR}
"
	exit ${1:-0}
}

if [[ "${#}" -eq "0" ]]; then
	_help 1
fi

function _info() {
	if [[ "${DRY_RUN}" ]]; then
		echo -e "${MAGENTA}[$(date +%F\ %T)] 🛟 DRY_RUN mode ${CLEAR}${1}${CLEAR}"
	else
		echo -e "${BLUE}[$(date +%F\ %T)] ${CLEAR}${1}${CLEAR}"
	fi
}

function _error() {
	echo -e "${RED}[$(date +%F\ %T)] ❌ ${1}${CLEAR}\n"
	_help 1
}

function divider() {
	[[ "${COLUMNS}" ]] || COLUMNS=80
	eval printf '=%.0s' {1..$COLUMNS}
	echo
}

function print_header() {
	echo -e "${CYAN}
   *@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*
 *@@@@@........@.@@.@.@@@.@@....@@@@@....@...@@@@@@@@@@@.@.@@...@@.@.@@@@@.@@........@@@@@*
@@@@..@@..@@@@@@@@@@......@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.@@@.....@@@@@@@@@@@@.@@.@@@@@
@@@.@..@@.@@@@.@@.@@@@@@@@@@..@...@......@..@..@......@..@..@@.@@@@@@@@@@@..@@@@@.@@..@@.@@@
@@.@...@..@@..@@@@...@.....@..@@.@.@@..@.@.@@..@..@@..@..@..@..@..@.....@.......@@.@@..@.@@@
@@.@...@..@@..@@@@..@..@@..@..@..@@@@..@@......@..@@..@..@..@..@@@@..@@@.@..@@..@@.@....@.@@
@@.@@@.@..@...@@@@..@..@...@.....@@@...@@@@@@..@..@@..@..@..@@...@@...@.@@...@@@@@.@.@@@.@@@
@@..@..@.@@..@@@@@..@..@@@.@..@..@@@...@@...@..@..@@..@..@@.@.@@@..@....@@@...@@@@.@..@@.@@@
@@@@.....@@..@@@@@..@..@...@..@..@@@..@@@...@..@......@........@@..@..@@@@@@@...@@.....@@@@@
@@@@@@@@.@@..@@@.@..@.....@...@.@@@@@@@@@@@@@@@@@@@@@@@@@@..@@.....@..@@.@...@@..@@.@@@@@@@@
 @@@@@@@.@...@@.....@@@@@@@@@@@@@@@@..@......@....@@..@@@@@@@@@@@@@@@....@.@.@@...@.@@@@@@@
  *@@@@@.@.@.....@@@@@@@...@.....@..@..@..@@...@@.@..@..@..@@.@....@@@@@@@@.......@.@@@@@*
 @@@@@..@@..@@@@@@@@@@..@@.@..@..@..@..@...@@..@@@@..@..@...@.@.@@..@@@@@@@@@@...@@.@@@@@@@
@@@@@@.@@@@@@@......@@..@@@@.@@..@..@..@..@@@..@@@@..@..@.....@@@..@@@.......@@@@@..@@@@@@@@
@@@@@@@@...@@@@@@@@.@@..@@@@.@@..@.@@..@..@@...@@.@..@@.@..@...@@..@@..@@@@@.@@@..@@@@@@@@@@
@@@@@@@@@@@@.....@@.@@.@@........@....@@.....@....@@....@..@@..@..@@@@.@@@.....@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@.@@....@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.@.....@.@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@.@@@@@@@@@@@..@....@@.@@@@@@@@@@....@@..@@@@@@@@@@@.@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@.@@@....@@@@@@@@@@...............@@@@@@@@@@@....@@@.@@@@@@@@@@@@@@@@@@@@@
 *@@@@@@@@@@@@@@@@@@@..@@@.....@@@@@@@@@@@@@@@@@@@@@@@@@@@@.....@@@...@@@@@@@@@@@@ ${YELLOW}2024${CYAN} @@*
   *@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*
  ${CLEAR}"
}

function load_config() {
	# ARGV1 = game-config
	unset BOT_LIST
	declare -a bots
	source "${1}" || _error "Can't load game-config ${YELLOW}${1}"
	if [[ ! -r "${MAPS_DIR}/${map}" ]]; then
		_error "${map} does not exist!"
	fi
	export THIS_MAP="${map}"
	_info "📝 Loaded configfile: ${YELLOW}${1}"
	_info "🗺️ Loaded game map: ${YELLOW}${map}"
	_info "🤖 Loaded ${#bots[@]} bots: ${YELLOW}${bots[*]}"
	# randomize bot list
	botnum=$((${#bots[@]} - 1))
	newseq=$(shuf -i 0-$botnum)
	for i in ${newseq}; do
		export BOT_LIST+=("${bots[$i]}")
	done
	# make sure log & output dir exists
	for dir in log output; do
		[ -d "${REPO_DIR}/${dir}" ] || mkdir -p "${REPO_DIR}/${dir}"
	done
	export GAME_CONFIG_FILE="${1}"
}

function create_docker_compose() {
	_info "🐳 ${BOLD}Creating docker compose file: ${YELLOW}${DOCKER_COMPOSE_FILE}"
	cat <<EOF >${DOCKER_COMPOSE_FILE}
name: game

networks:
  ${DOCKER_NETWORK}:

services:
EOF
}

function build_game_server() {
	_info "👷🏻‍♂️ Building game server"
	docker build -f ${DOCKERFILE_GAME} . -t ${GAME_SERVER_NAME} &>/dev/null || _error "Something went wrong whithin function ${FUNCNAME}"
}

function add_game_server() {
	if (docker image inspect ${GAME_SERVER_NAME} &>/dev/null); then
		if [[ ${REBUILD_SERVER} ]]; then
			_info "🚩 force REBUILD_SERVER was set"
			docker rmi -f $(docker images | awk "/^${GAME_SERVER_NAME}/ {print \$3}")
			build_game_server
		fi
	else
		build_game_server
	fi
	# add the game-server to docker-compose file
	# we send the map via GAME.BOARD_PATH env var
	cat <<EOF >>${DOCKER_COMPOSE_FILE}

  ${GAME_SERVER_NAME}:
    image: ${GAME_SERVER_NAME}
    container_name: ${GAME_SERVER_NAME}
    hostname: ${GAME_SERVER_NAME}
    restart: no
    environment:
      GAME.BOARD_PATH: "/maps/${THIS_MAP}"
    volumes:
      - ${MAPS_DIR}:/maps:ro
      - ${OUTPUT_DIR}:/app/output:rw
    ports:
      - ${GAME_SERVER_PORT}
    networks:
      - ${DOCKER_NETWORK}
EOF
	_info "🚥 ${GREEN}Added game server"
}

function pull_image() {
	# pull the bot image from public registry
	[ "${DONT_PULL}" ] || docker pull "${1}" || _error "Cannot pull image ${YELLOW}${1}"
}

function add_all_bots() {
	# calls add_bot for each participant defined in game-config
	for i in ${BOT_LIST[@]}; do
		add_bot "${i}"
	done
}

function add_bot() {
	# add the bot to docker-compose file
	local THIS_IMAGE="${1}"
	if [ ! "${DRY_RUN}" ]; then
		_info "⏬ ${BOLD}Pulling container image ${YELLOW}${THIS_IMAGE} ..."
		pull_image "${THIS_IMAGE}" || _error "Something went wrong in ${FUNCNAME}: error while pulling image ${YELLOW}${THIS_IMAGE}"
	fi
	local THIS_BOT_NAME="$(echo "${THIS_IMAGE}" | awk -F'/' '{print $2"-"$3}')"
	cat <<EOF >>${DOCKER_COMPOSE_FILE}

  ${THIS_BOT_NAME}:
    environment:
      BOT_NAME: ${THIS_BOT_NAME}
    image: ${THIS_IMAGE}  
    container_name: ${THIS_BOT_NAME}
    hostname: ${THIS_BOT_NAME}
    restart: no
    ports:
      - ${BOT_DEFAULT_PORT}
    networks:
      - ${DOCKER_NETWORK}
    depends_on:
      ${GAME_SERVER_NAME}:
        condition: service_started
EOF
	_info "🤖 ${GREEN}Added ${YELLOW}${THIS_BOT_NAME}${CLEAR} from ${YELLOW}${THIS_IMAGE}"
}

function create_simulation() {
	# creates a e2e simulation with 6 fake registries/projects/bots
	_info "‼️ ${YELLOW}Entering ${FUNCNAME}"
	export TMP_CONFIG="${TMPDIR}game.cfg"
	declare -a bots
	declare -a maps
	local bots=('docker.io/fernando/bot' 'quay.io/shresth/bot' 'quay.io/yvette/bot' 'ghcr.io/amadis/bot' 'quay.io/loren/bot' 'ghcr.io/jose/bot')
	local maps=(${REPO_DIR}/maps/*)
	local random_map="$(basename $(printf "%s\n" "${maps[@]}" | shuf -n 1))"
	echo -e "bots=('docker.io/fernando/bot' 'quay.io/shresth/bot' 'quay.io/yvette/bot' 'ghcr.io/amadis/bot' 'quay.io/loren/bot' 'ghcr.io/jose/bot')\nmap=${random_map}" >${TMP_CONFIG}
	_info "📒 Using tmp game-config:\n$(cat "${TMP_CONFIG}")"
	_info "🗺️ Using random map ${random_map}:\n$(cat "${MAPS_DIR}/${random_map}")"

	echo
	REPO_DIR="$(git rev-parse --show-toplevel)"

	function create_dockerfile() {
		local BOT_NAME="$2"
		cd "${REPO_DIR}" || _error "Error entering dir [${REPO_DIR}] in function ${FUNCNAME} while creating bot [${BOT_NAME}] image!"
		cat <<EOF >${1}
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY . .
RUN go mod download && CGO_ENABLED=0 GOOS=linux go build -o bot ./examples/ranbot.go

FROM alpine:3.20.3
WORKDIR /app
COPY ./proto/ ./proto/
COPY --from=builder /app/bot ./
RUN adduser -h /app -H -s /sbin/nologin -D -u 10000 bot-user && chown -R bot-user:bot-user /app
USER bot-user
EXPOSE 3001
CMD [ "./bot", "-bn=${BOT_NAME}", "-la=${BOT_NAME}:3001", "-gs=${GAME_SERVER_NAME}:${GAME_SERVER_PORT}" ]
EOF
	}

	for i in "${bots[@]}"; do
		THIS_NAME="$(echo "${i}" | awk -F'/' '{print $2"-"$3}')"
		THIS_DOCKERFILE="${TMPDIR}Dockerfile-${RANDOM}"
		create_dockerfile "${THIS_DOCKERFILE}" "${THIS_NAME}"
		_info "🐳 Building ${YELLOW}${i}${CLEAR} from ${CYAN}\$TMPDIR/$(basename ${THIS_DOCKERFILE})${CLEAR}\t"
		local DOCKER_BUILD_COMMAND="docker build -f ${THIS_DOCKERFILE} . -t ${i}:latest &>/dev/null"
		eval ${DOCKER_BUILD_COMMAND}
		if [[ "$?" -ne "0" ]]; then
			_error "Something went wrong while building docker image with Dockerfile ${THIS_DOCKERFILE}\nCommand:  ${DOCKER_BUILD_COMMAND}\nDockerfile:"
			cat "${THIS_DOCKERFILE}"
			rm -f "${THIS_DOCKERFILE}"
			rm -f "${TMP_CONFIG}"
			echo "❌"
			cleanup
			exit 1
		fi
		rm -f "${THIS_DOCKERFILE}"
		unset THIS_REGISTRY
		unset THIS_NAME
		unset THIS_DOCKERFILE
	done
	# load the DRY_RUN game-config
	load_config "${TMP_CONFIG}" || destroy_simulation
}

function destroy_simulation() {
	_info "🐲 ${YELLOW}Entering ${FUNCNAME}"
	source "${TMP_CONFIG}"
	for i in ${bots[*]}; do
		docker ps -a | grep '/bot' | awk '{print $1}' | while read foo; do
			docker stop $foo &>/dev/null
			docker rm $foo &>/dev/null
		done
		docker images | grep '/bot' | awk '{print $3}' | xargs --no-run-if-empty docker rmi -f &>/dev/null &&
			_info "  🔥 ${YELLOW}${i}${CLEAR}\thas gone bye bye"
	done
	rm -f "${TMP_CONFIG}"
}

function cleanup() {
	_info "🧹 ${GREEN}Cleaning up..."
	docker compose -f ${DOCKER_COMPOSE_FILE} down 2>/dev/null
	docker ps -a --format "{{.Names}}" | egrep -E '^(game|bot)' | xargs --no-run-if-empty docker stop 2>/dev/null
	rm -f "${DOCKER_COMPOSE_FILE}"
	if [[ "${DRY_RUN}" ]]; then
		destroy_simulation
	fi
}

###############################################################################
# 1. validate script args

while getopts df:rxh Option; do
	case "${Option}" in
	d)
		export DRY_RUN=1
		;;
	f)
		if [[ ! "${DRY_RUN}" ]]; then
			load_config "${OPTARG}"
		fi
		;;
	r)
		export REBUILD_SERVER=1
		;;
	x)
		# dirty, I know..
		export DONT_PULL=1
		;;
	*)
		_help 0
		;;
	esac
done

###############################################################################
# 2. prepare

[[ "${DRY_RUN}" ]] && create_simulation
print_header
create_docker_compose
add_all_bots
add_game_server

###############################################################################
# 3. start

# initialize log & game round
COMMAND_UP="docker compose -f ${DOCKER_COMPOSE_FILE} up --timestamps --abort-on-container-exit"

(
	cat <<EOF
# ::: CodeconZ 2024 ::: LighthouseS AI Contest :::
#
# Config file: ${GAME_CONFIG_FILE}
$(cat "${GAME_CONFIG_FILE}")

# Map: ${GAME_CONFIG_FILE}

EOF
	_info "🚀 ${GREEN}Launching new round on $(date +%F\ %T)"
	eval ${COMMAND_UP}
) | tee "${LOG_FILE}"

###############################################################################
# 4. cleanup

divider
cleanup
divider
_info "📝 Log file:    ${CYAN}./log/$(basename "${LOG_FILE}")"
_info "📊 Game Output: ${GREEN}$(ls -1t ./$(basename ${OUTPUT_DIR})/*.json | head -1)"
divider
_info "✅ ${GREEN}All done, bye 👋🏻"
echo

# EOF
