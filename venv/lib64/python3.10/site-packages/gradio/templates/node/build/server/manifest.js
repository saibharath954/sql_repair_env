const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {start:"_app/immutable/entry/start.KCY16dCL.js",app:"_app/immutable/entry/app.D9WUA4GP.js",imports:["_app/immutable/entry/start.KCY16dCL.js","_app/immutable/chunks/C_Xa3Bt6.js","_app/immutable/chunks/DD_ROOFA.js","_app/immutable/chunks/BzNkWfQm.js","_app/immutable/entry/app.D9WUA4GP.js","_app/immutable/chunks/DAZnPtxU.js","_app/immutable/chunks/DD_ROOFA.js","_app/immutable/chunks/BzNkWfQm.js","_app/immutable/chunks/DNgPkysA.js","_app/immutable/chunks/DGd53R1Z.js"],stylesheets:[],fonts:[],uses_env_dynamic_public:false},
		nodes: [
			__memo(() => import('./chunks/0-BSwhZmkj.js')),
			__memo(() => import('./chunks/1-D-7qZIDe.js')),
			__memo(() => import('./chunks/2-DKaY_6dX.js').then(function (n) { return n._; }))
		],
		remotes: {
			
		},
		routes: [
			{
				id: "/[...catchall]",
				pattern: /^(?:\/([^]*))?\/?$/,
				params: [{"name":"catchall","optional":false,"rest":true,"chained":true}],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		prerendered_routes: new Set([]),
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();

const prerendered = new Set([]);

const base = "";

export { base, manifest, prerendered };
//# sourceMappingURL=manifest.js.map
