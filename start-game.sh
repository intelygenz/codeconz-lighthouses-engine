#!/usr/bin/env bash
###############################################################################
#               ::: CodeconZ 2024 ::: LighthouseS AI Contest :::              #
###############################################################################
#
# For usage instructions run:
#   ./start-game.sh -h
#
###############################################################################

# Configuration
BOT_DEFAULT_PORT='3001'
GAME_NETWORK_NAME='game_net'
GAME_IMAGE_NAME='intelygenz/codeconz-lighthouses-engine'
GAME_CONTAINER_NAME='game'
GAME_SERVER_PORT='50051'

###############################################################################

REPO_DIR="$(git rev-parse --show-toplevel)"
MAPS_DIR="${REPO_DIR}/maps"
LOG_DIR="${REPO_DIR}/log"
OUTPUT_DIR="${REPO_DIR}/output"
GAME_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
DOCKERFILE_GAME="${REPO_DIR}/Dockerfile"
DOCKER_COMPOSE_FILE="game-${GAME_TIMESTAMP}.yaml"
LOG_FILE="${LOG_DIR}/game-${GAME_TIMESTAMP}.log"
COMMAND_UP="docker compose -f ${DOCKER_COMPOSE_FILE} up --timestamps --abort-on-container-exit"
export DOCKER_DEFAULT_PLATFORM=linux/amd64

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
  ./$(basename "${0}") [-x][-r] -f <configfile>   ${MAGENTA}run a game${CLEAR}

Options:

  -r    force rebuild the game server docker image (optional)
  -x    do not pull docker images (use local ones)
  -f    specify game-config file (${MAGENTA}REQUIRED${CLEAR})

  The game-config file format must be:

    # An array of bot images that will play the game.
    # Each element in the array needs to be a docker pull URI for a PUBLIC image.
    # A ':latest' tag will always be used when pulling images.
    bots=(${YELLOW}'ghcr.io/john/bot-foo' 'docker.io/jane/bot-bar' ... 'quay.io/dave/bot-baz'${CLEAR})

    # The map file that will be used in the game.
    # The map file MUST exist in the ./maps/ folder.
    map=${YELLOW}square.txt${CLEAR}

    # The number of turns the game will last.
    turns=${YELLOW}500${CLEAR}

    # The time the engine will wait for a bot to respond to a turn request.
    turn_request_timeout=${YELLOW}100ms${CLEAR}

    # The time the engine will wait between rounds.
    time_between_rounds=${YELLOW}0s${CLEAR}
"
	exit ${1:-0}
}

if [[ "${#}" -eq "0" ]]; then
	_help 1
fi

function _info() {
	echo -e "${BLUE}[$(date +%F\ %T)] ${CLEAR}${1}${CLEAR}"
}

function _error() {
	echo -e "${RED}[$(date +%F\ %T)] ❌ ${1}${CLEAR}\n"
	exit 1
}

# check required tools
(docker compose version &>/dev/null) || _error "Install docker compose first.${CLEAR}\n\thttps://docs.docker.com/compose/install/"
(type shuf &>/dev/null) || _error "Install 'coreutils' package first (required: shuf)."
(type awk &>/dev/null) || _error "Install 'awk' package first."

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
	export THIS_GAME_CONFIG="${1}"

	unset BOT_LIST bots newseq
	declare -a bots
	declare -a newseq
	source "${1}" || _error "Can't load game-config ${YELLOW}${1}"
	if [[ ! -r "${MAPS_DIR}/${map}" ]]; then
		_error "Map file '$(basename "${MAPS_DIR}")/${map}' not found. Check your configuration: ${1}"
	fi
	export THIS_MAP="${map}"

	# check if turns is defined
	if [[ ! "${turns}" ]]; then
		_error "Missing 'turns' from configuration file"
	fi
	export THIS_GAME_TURNS="${turns}"

	# check if turn_request_timeout is defined
	if [[ ! "${turn_request_timeout}" ]]; then
		_error "Missing 'turn_request_timeout' from configuration file"
	fi
	export THIS_GAME_REQ_TIMEOUT="${turn_request_timeout}"

	# check if time_between_rounds is defined
	if [[ ! "${time_between_rounds}" ]]; then
		_error "Missing 'time_between_rounds' from configuration file"
	fi
	export THIS_GAME_TIME_ROUNDS="${time_between_rounds}"

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
	# fix output dir permissions *required*
	chmod 0777 "${OUTPUT_DIR}"
	export GAME_CONFIG_FILE="${1}"
}

function print_config() {
	_info "📔 ${GREEN}Loaded configfile: ${YELLOW}${THIS_GAME_CONFIG}"
	_info "   bots: ${YELLOW}${BOT_LIST[*]}"
	_info "   map: ${YELLOW}${THIS_MAP}"
	_info "   turns: ${YELLOW}${THIS_GAME_TURNS}"
	_info "   turn_request_timeout: ${YELLOW}${THIS_GAME_REQ_TIMEOUT}"
	_info "   time_between_rounds: ${YELLOW}${THIS_GAME_TIME_ROUNDS}"
}

function create_docker_compose() {
	_info "🐳 ${GREEN}Creating docker compose file: ${YELLOW}${DOCKER_COMPOSE_FILE}"
	cat <<EOF >${DOCKER_COMPOSE_FILE}
name: game

networks:
  ${GAME_NETWORK_NAME}:

services:
EOF
}

function build_game_server() {
	_info "  👷 ${GREEN}Building ${YELLOW}${GAME_IMAGE_NAME}"
	docker build -f ${DOCKERFILE_GAME} . -t ${GAME_IMAGE_NAME} &>/dev/null || _error "Something went wrong whithin function ${FUNCNAME}"
}

function add_game_server() {
	if (docker image inspect ${GAME_IMAGE_NAME} &>/dev/null); then
		if [[ ${REBUILD_SERVER} ]]; then
			_info "  🚩 ${RED}force REBUILD_SERVER was set"
			docker rmi -f $(docker images | awk "/^${GAME_IMAGE_NAME//\//\\/}/ {print \$3}") 1>/dev/null
			build_game_server
		fi
	else
		build_game_server
	fi
	# add the game-server to docker-compose file
	cat <<EOF >>${DOCKER_COMPOSE_FILE}

  ${GAME_CONTAINER_NAME}:
    image: ${GAME_IMAGE_NAME}
    container_name: ${GAME_CONTAINER_NAME}
    hostname: ${GAME_CONTAINER_NAME}
    restart: no
    environment:
      BOARD_PATH: "/maps/${THIS_MAP}"
      TURNS: "${THIS_GAME_TURNS}"
      TURN_REQUEST_TIMEOUT: "${THIS_GAME_REQ_TIMEOUT}"
      TIME_BETWEEN_ROUNDS: "${THIS_GAME_TIME_ROUNDS}"
    volumes:
      - ${MAPS_DIR}:/maps:ro
      - ${OUTPUT_DIR}:/app/output:rw
    ports:
      - ${GAME_SERVER_PORT}
    networks:
      - ${GAME_NETWORK_NAME}
EOF
	_info "  🧠 ${GREEN}Added ${YELLOW}${GAME_IMAGE_NAME}${CLEAR} as ${YELLOW}${GAME_CONTAINER_NAME}"
}

function pull_image() {
	# pull the bot image from public registry
	if [[ "${DEBUG}" ]]; then
		docker pull "${1}" || _error "Cannot pull image ${YELLOW}${1}"
	else
		docker pull "${1}" &>/dev/null || _error "Cannot pull image ${YELLOW}${1}"
	fi
}

function add_all_bots() {
	# calls add_bot for each participant defined in game-config
	for i in "${BOT_LIST[@]}"; do
		add_bot "${i}"
	done
}

function add_bot() {
	# add the bot to docker-compose file
	local THIS_IMAGE="${1}"
	if [ ! "${DONT_PULL}" ]; then
		_info "  ⏬ ${BOLD}Pulling container image ${YELLOW}${THIS_IMAGE} ..."
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
      - ${GAME_NETWORK_NAME}
    depends_on:
      ${GAME_CONTAINER_NAME}:
        condition: service_started
EOF
	_info "  🤖 ${GREEN}Added ${YELLOW}${THIS_IMAGE}${CLEAR} as ${YELLOW}${THIS_BOT_NAME}"
}

function cleanup() {
	_info "🧹 ${GREEN}Cleaning up..."
	docker compose -f ${DOCKER_COMPOSE_FILE} down 2>/dev/null
	docker ps -a --format "{{.Names}}" | grep -E '^(engine|bot)' | xargs --no-run-if-empty docker stop 2>/dev/null
	[ "${DEBUG}" ] || rm -f "${DOCKER_COMPOSE_FILE}"
}

function create_game_log() {
	cat <<EOF >"${LOG_FILE}"
# :::
# ::: CodeconZ 2024
# ::: LighthouseS AI Contest
# :::
# ::: Game:        ${GAME_TIMESTAMP}
# ::: Config file: ${GAME_CONFIG_FILE}
# ::: Players:     $(echo "${BOT_LIST[*]}" | xargs)
# ::: Map:         $(echo "${THIS_MAP}")
# :::

EOF
}

function create_player_log() {
	for i in "${BOT_LIST[@]}"; do
		local THIS_BOT_NAME="$(echo "${i}" | awk -F'/' '{print $2"-"$3}')"
		local PLAYER_LOG_FILE="${LOG_DIR}/game-${GAME_TIMESTAMP}__${THIS_BOT_NAME}.log"
		cat <<EOF >"${PLAYER_LOG_FILE}"
# :::
# ::: CodeconZ 2024
# ::: LighthouseS AI Contest
# :::
# ::: Game:       ${GAME_TIMESTAMP}
# ::: Player:     ${THIS_BOT_NAME}
# :::

EOF
		grep -E -i "game starts|game finished|${THIS_BOT_NAME}" "${LOG_FILE}" | grep -v '^Attaching' >>"${PLAYER_LOG_FILE}"
		_info "📝 Created player log file: ${GREEN}$(basename "${LOG_DIR}")/$(basename "${PLAYER_LOG_FILE}")"
	done
}

###############################################################################
# 1. validate script args

while getopts rxf:h Option; do
	case "${Option}" in
	r)
		export REBUILD_SERVER=1
		;;
	x)
		export DONT_PULL=1
		;;
	f)
		load_config "${OPTARG}"
		;;
	*)
		_help 0
		;;
	esac
done

###############################################################################
# 2. prepare

divider
print_config
divider
_info "👷 ${GREEN}Setting up game and bots"
create_docker_compose
add_all_bots
add_game_server

###############################################################################
# 3. start

# initialize log & game round
divider
_info "🚀 ${GREEN}Starting game!"
create_game_log
print_header
(
	export DOCKER_DEFAULT_PLATFORM=linux/amd64
	eval "${COMMAND_UP}"
) | tee -a "${LOG_FILE}"

###############################################################################
# 4. cleanup

divider
cleanup
_info "🏁 ${GREEN}Game ended!"
_info "📝 Log file:    ${CYAN}$(basename "${LOG_DIR}")/$(basename "${LOG_FILE}")"

GAME_OUTPUT_JSON="$(ls -1t $(basename "${OUTPUT_DIR}")/*.json 2>/dev/null | head -1)"
if [ -s "${GAME_OUTPUT_JSON}" ]; then
	_info "📊 Game Output: ${GREEN}${GAME_OUTPUT_JSON}"
	create_player_log
	divider
	_info "✅ ${GREEN}All done, may the force be with you!"
else
	_error "Something went wrong... 💥"
fi

echo

# EOF
