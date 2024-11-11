export const init = async (container, playback) => {
  const app = new pixi.Application();
  await app.init({ resizeTo: container });

  container.appendChild(app.canvas);

  const appWidth = app.renderer.width;
  const appHeight = app.renderer.height;

  const gridWidth = game.board.tiles[0].length;
  const gridHeight = game.board.tiles.length;
  const maxWidth = (appWidth - 100) / gridWidth;
  const maxHeight = (appHeight - 100) / gridHeight;
  const tileSize = Math.min(maxWidth, maxHeight);
  const x = appWidth / 2 - (gridWidth * tileSize) / 2;
  const y = appHeight / 2 - (gridHeight * tileSize) / 2;

  const board = new Board(game, tileSize, x, y);
  app.stage.addChild(board);

  const border = new pixi.Graphics();
  border
    .rect(x, y, tileSize * gridWidth, tileSize * gridHeight)
    .stroke({ color: 0x606060, width: 1 });
  app.stage.addChild(border);

  const ticker = pixi.Ticker.shared;
  ticker.autoStart = false;
  playback.init(
    (frame) => board.render(frame),
    () => ticker.start(),
    () => ticker.stop(),
  );

  ticker.add(() => {
    if (playback.status() === PlaybackStatus.done) {
      ticker.stop();
    }
  });
};
