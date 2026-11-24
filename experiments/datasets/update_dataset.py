import os
import json

DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(DATASETS_DIR, "cache")

REPO_MAP = {
    "pallets/flask": "flask",
    "expressjs/express": "express",
    "fmtlib/fmt": "fmt"
}

# Complete 60 questions definition with:
# 1. Natural context-dependent Turn-2 follow-ups
# 2. Multi-file evidence annotations
# 3. Stripped leaked file paths & balanced conceptual vs identifier queries
# 4. Explicit query_style attribute
UPDATED_BENCHMARK = [
    # =========================================================================
    # PALLETS/FLASK (20 questions: flask_001 to flask_020)
    # =========================================================================
    {
        "question_id": "flask_001",
        "repository": "pallets/flask",
        "category": "function_identification",
        "query_style": "identifier",
        "question": "Where is the Flask application class defined and what parameters does its __init__ constructor accept?",
        "conversation_id": "flask_conv_01",
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query testing locating the core application class definition.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/app.py",
                "symbol": "class Flask",
                "start_line": 108,
                "end_line": 140,
                "relevance_reason": "Defines the core Flask application class subclassing App and documents its initialization parameters."
            }
        ]
    },
    {
        "question_id": "flask_002",
        "repository": "pallets/flask",
        "category": "implementation_understanding",
        "query_style": "conversational",
        "question": "How does it configure default configuration settings during that initialization process?",
        "conversation_id": "flask_conv_01",
        "turn_number": 2,
        "previous_context": "In flask_001, we asked where the Flask class is defined and what its constructor accepts.",
        "expected_standalone_query": "How does the Flask application initialize and configure its default configuration settings during initialization?",
        "notes": "Context-dependent conversational query using 'it' and 'that initialization process'.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/sansio/app.py",
                "symbol": "make_config",
                "start_line": 480,
                "end_line": 505,
                "relevance_reason": "Implements make_config which creates a Config object and loads default_config into it."
            }
        ]
    },
    {
        "question_id": "flask_003",
        "repository": "pallets/flask",
        "category": "function_identification",
        "query_style": "identifier",
        "question": "Where is the Request wrapper class defined in Flask and what base class does it inherit from?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query locating the Request wrapper class.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/wrappers.py",
                "symbol": "class Request",
                "start_line": 18,
                "end_line": 45,
                "relevance_reason": "Defines Flask's Request class subclassing Werkzeug RequestBase and setting json module attributes."
            }
        ]
    },
    {
        "question_id": "flask_004",
        "repository": "pallets/flask",
        "category": "function_identification",
        "query_style": "identifier",
        "question": "Where is the url_for helper function defined and how does it generate URLs from endpoints?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query locating the public url_for function.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/helpers.py",
                "symbol": "url_for",
                "start_line": 200,
                "end_line": 245,
                "relevance_reason": "Implements the public url_for function that queries the active RequestContext and URL adapter."
            }
        ]
    },
    {
        "question_id": "flask_005",
        "repository": "pallets/flask",
        "category": "implementation_understanding",
        "query_style": "identifier",
        "question": "How does Flask's route decorator register an endpoint and view function onto the application?",
        "conversation_id": "flask_conv_02",
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Root query on routing decorator implementation.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/sansio/scaffold.py",
                "symbol": "def route",
                "start_line": 344,
                "end_line": 370,
                "relevance_reason": "Defines Scaffold.route decorator that returns a wrapper calling add_url_rule."
            }
        ]
    },
    {
        "question_id": "flask_006",
        "repository": "pallets/flask",
        "category": "cross_file_dependencies",
        "query_style": "conversational",
        "question": "How does that method configure those routing rules and delegate to the underlying routing Map?",
        "conversation_id": "flask_conv_02",
        "turn_number": 2,
        "previous_context": "In flask_005, we examined how the route decorator delegates to add_url_rule.",
        "expected_standalone_query": "How does Scaffold.add_url_rule configure routing rules and delegate to Werkzeug's routing Map in Flask?",
        "notes": "Context-dependent query with multi-file evidence across scaffold.py and app.py.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/sansio/scaffold.py",
                "symbol": "add_url_rule",
                "start_line": 376,
                "end_line": 425,
                "relevance_reason": "Shows how add_url_rule constructs Rule instances, validates endpoints, and registers view functions."
            },
            {
                "file_path": "src/flask/app.py",
                "symbol": "create_url_adapter",
                "start_line": 507,
                "end_line": 530,
                "relevance_reason": "Shows how the Flask application creates the URL adapter from url_map to bind rules to the request."
            }
        ]
    },
    {
        "question_id": "flask_007",
        "repository": "pallets/flask",
        "category": "implementation_understanding",
        "query_style": "identifier",
        "question": "How does dispatch_request in Flask execute the endpoint matched from the active request URL?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query inspecting dispatch_request.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/app.py",
                "symbol": "dispatch_request",
                "start_line": 993,
                "end_line": 1025,
                "relevance_reason": "Shows fetching the view function from view_functions for req.endpoint and invoking it with view_args."
            }
        ]
    },
    {
        "question_id": "flask_008",
        "repository": "pallets/flask",
        "category": "cross_file_dependencies",
        "query_style": "conceptual",
        "question": "How does the framework isolate request and application state across threads and push contexts onto the active stack?",
        "conversation_id": "flask_conv_03",
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual multi-file query on Flask context isolation without naming exact class names.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/ctx.py",
                "symbol": "class AppContext",
                "start_line": 260,
                "end_line": 305,
                "relevance_reason": "Defines AppContext class, associating it with the Flask app, and managing the _cv_app context variable."
            },
            {
                "file_path": "src/flask/app.py",
                "symbol": "def app_context",
                "start_line": 1542,
                "end_line": 1575,
                "relevance_reason": "Implements app_context and request_context factory methods on the Flask application instance."
            }
        ]
    },
    {
        "question_id": "flask_009",
        "repository": "pallets/flask",
        "category": "architecture_flow",
        "query_style": "conversational",
        "question": "Walk through what happens inside the application entrypoint when an incoming request is pushed through those contexts.",
        "conversation_id": "flask_conv_03",
        "turn_number": 2,
        "previous_context": "In flask_008, we examined context stack isolation and AppContext creation.",
        "expected_standalone_query": "Walk through the execution flow inside wsgi_app when an incoming WSGI HTTP request arrives and pushes contexts in Flask.",
        "notes": "Context-dependent query with multi-file evidence tracing app.py wsgi_app and ctx.py push.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/app.py",
                "symbol": "wsgi_app",
                "start_line": 1627,
                "end_line": 1670,
                "relevance_reason": "Implements the core WSGI entrypoint that creates request_context, pushes it, dispatches the request, and handles errors."
            },
            {
                "file_path": "src/flask/ctx.py",
                "symbol": "def push(",
                "start_line": 416,
                "end_line": 445,
                "relevance_reason": "Implements AppContext.push binding the context to the active context variable token."
            }
        ]
    },
    {
        "question_id": "flask_010",
        "repository": "pallets/flask",
        "category": "function_identification",
        "query_style": "identifier",
        "question": "Where is the Blueprint class defined and what are its initialization parameters?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query locating Blueprint class definition.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/sansio/blueprints.py",
                "symbol": "class Blueprint",
                "start_line": 119,
                "end_line": 155,
                "relevance_reason": "Defines the Blueprint class subclassing Scaffold, taking name and import_name parameters."
            }
        ]
    },
    {
        "question_id": "flask_011",
        "repository": "pallets/flask",
        "category": "cross_file_dependencies",
        "query_style": "conceptual",
        "question": "How do modular blueprints record deferred route handlers and apply them to the main application during registration?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual multi-file query on blueprint registration mechanics across blueprints.py and app.py.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/sansio/blueprints.py",
                "symbol": "register",
                "start_line": 273,
                "end_line": 315,
                "relevance_reason": "Implements Blueprint.register which executes recorded deferred functions using BlueprintSetupState."
            },
            {
                "file_path": "src/flask/sansio/app.py",
                "symbol": "def register_blueprint",
                "start_line": 571,
                "end_line": 605,
                "relevance_reason": "Implements register_blueprint on the application, checking for existing blueprints and calling blueprint.register."
            }
        ]
    },
    {
        "question_id": "flask_012",
        "repository": "pallets/flask",
        "category": "architecture_flow",
        "query_style": "conceptual",
        "question": "How does Flask handle unhandled server exceptions and transform them into HTTP 500 error responses?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual query describing 500 exception handling lifecycle.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/app.py",
                "symbol": "handle_exception",
                "start_line": 916,
                "end_line": 955,
                "relevance_reason": "Shows how exceptions trigger got_request_exception signal, error handler lookup, and fallback 500 responses."
            }
        ]
    },
    {
        "question_id": "flask_013",
        "repository": "pallets/flask",
        "category": "architecture_flow",
        "query_style": "identifier",
        "question": "Where is the session interface implemented and how does SecureCookieSessionInterface open and save sessions?",
        "conversation_id": "flask_conv_04",
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Root query on session interface with multi-file evidence connecting sessions.py and app.py.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/sessions.py",
                "symbol": "class SecureCookieSessionInterface",
                "start_line": 284,
                "end_line": 340,
                "relevance_reason": "Defines SecureCookieSessionInterface with its itsdangerous serializer, open_session, and save_session implementations."
            },
            {
                "file_path": "src/flask/app.py",
                "symbol": "session_interface",
                "start_line": 245,
                "end_line": 260,
                "relevance_reason": "Attaches SecureCookieSessionInterface as the default session_interface on the Flask application class."
            }
        ]
    },
    {
        "question_id": "flask_014",
        "repository": "pallets/flask",
        "category": "debugging_configuration",
        "query_style": "conversational",
        "question": "How does it configure security flags like HttpOnly and Secure on those session cookies?",
        "conversation_id": "flask_conv_04",
        "turn_number": 2,
        "previous_context": "In flask_013, we examined how SecureCookieSessionInterface handles sessions.",
        "expected_standalone_query": "How does Flask configure cookie security flags like SESSION_COOKIE_SECURE and SESSION_COOKIE_HTTPONLY in SessionInterface?",
        "notes": "Context-dependent conversational query using 'it' and 'those session cookies'.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/sessions.py",
                "symbol": "get_cookie_httponly",
                "start_line": 195,
                "end_line": 225,
                "relevance_reason": "Shows getter methods on SessionInterface reading SESSION_COOKIE_HTTPONLY, SESSION_COOKIE_SECURE, and SESSION_COOKIE_SAMESITE."
            }
        ]
    },
    {
        "question_id": "flask_015",
        "repository": "pallets/flask",
        "category": "architecture_flow",
        "query_style": "conceptual",
        "question": "How does Flask coerce different controller return types such as strings, dictionaries, and tuples into WSGI responses?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual query on response conversion.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/app.py",
                "symbol": "make_response",
                "start_line": 1267,
                "end_line": 1315,
                "relevance_reason": "Implements Flask.make_response unpackaging status, headers, and converting strings/dicts to Response instances."
            }
        ]
    },
    {
        "question_id": "flask_016",
        "repository": "pallets/flask",
        "category": "implementation_understanding",
        "query_style": "identifier",
        "question": "How does Config.from_object in Flask inspect and load configuration values from a Python module or class?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query inspecting from_object.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/config.py",
                "symbol": "from_object",
                "start_line": 218,
                "end_line": 250,
                "relevance_reason": "Shows how from_object inspects uppercase attributes on an object or imports string module paths."
            }
        ]
    },
    {
        "question_id": "flask_017",
        "repository": "pallets/flask",
        "category": "debugging_configuration",
        "query_style": "conceptual",
        "question": "How does Flask bind Click command-line interface commands to an active application environment?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual query describing CLI context binding without naming with_appcontext.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/cli.py",
                "symbol": "with_appcontext",
                "start_line": 380,
                "end_line": 410,
                "relevance_reason": "Shows how with_appcontext wraps click commands and calls ensure_sync inside an app.app_context() block."
            }
        ]
    },
    {
        "question_id": "flask_018",
        "repository": "pallets/flask",
        "category": "debugging_configuration",
        "query_style": "conceptual",
        "question": "How does Flask pass temporary one-off notification messages between redirected requests?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual query describing flash messaging behavior without naming flash/get_flashed_messages.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "src/flask/helpers.py",
                "symbol": "flash",
                "start_line": 326,
                "end_line": 365,
                "relevance_reason": "Shows flash appending message tuples to session['_flashes'] and get_flashed_messages popping them."
            }
        ]
    },
    {
        "question_id": "flask_019",
        "repository": "pallets/flask",
        "category": "unanswerable",
        "query_style": "unanswerable",
        "question": "Where is the built-in database connection pool for PostgreSQL configured in Flask?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Unanswerable: Flask does not contain a built-in database connection pool.",
        "answerable": False,
        "evidence": [],
        "justification": "Flask does not include a built-in database connection pool or ORM. Database connectivity is handled by third-party extensions such as Flask-SQLAlchemy or external drivers (e.g. psycopg2)."
    },
    {
        "question_id": "flask_020",
        "repository": "pallets/flask",
        "category": "unanswerable",
        "query_style": "unanswerable",
        "question": "How does Flask configure native WebSocket subscriptions and publish-subscribe channels?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Unanswerable: Flask is a WSGI framework and lacks native WebSocket pub/sub channels.",
        "answerable": False,
        "evidence": [],
        "justification": "Core Flask is a WSGI synchronous web framework and does not provide native WebSocket protocol handlers or pub/sub channels. Real-time WebSocket capabilities require extensions such as Flask-SocketIO or ASGI frameworks like Quart."
    },

    # =========================================================================
    # EXPRESSJS/EXPRESS (20 questions: express_001 to express_020)
    # =========================================================================
    {
        "question_id": "express_001",
        "repository": "expressjs/express",
        "category": "function_identification",
        "query_style": "identifier",
        "question": "Where is createApplication defined in Express and how does it mix in EventEmitter and the application prototype?",
        "conversation_id": "express_conv_01",
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query with multi-file evidence across express.js and application.js.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/express.js",
                "symbol": "createApplication",
                "start_line": 36,
                "end_line": 57,
                "relevance_reason": "Defines createApplication factory function mixing in EventEmitter, application proto, and request/response prototypes."
            },
            {
                "file_path": "lib/application.js",
                "symbol": "app.init",
                "start_line": 59,
                "end_line": 83,
                "relevance_reason": "Implements app.init called by createApplication to initialize caches, engines, settings, and defaultConfiguration."
            }
        ]
    },
    {
        "question_id": "express_002",
        "repository": "expressjs/express",
        "category": "implementation_understanding",
        "query_style": "conversational",
        "question": "How does that created application set up its default configuration settings like etag, env, and query parser?",
        "conversation_id": "express_conv_01",
        "turn_number": 2,
        "previous_context": "In express_001, we saw how createApplication calls app.init().",
        "expected_standalone_query": "How does an Express application initialize default settings such as etag, env, and query parser in defaultConfiguration?",
        "notes": "Context-dependent conversational query using 'that created application'.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/application.js",
                "symbol": "app.defaultConfiguration",
                "start_line": 90,
                "end_line": 141,
                "relevance_reason": "Sets default environment, etag, query parser, trust proxy, and view settings in app.defaultConfiguration."
            }
        ]
    },
    {
        "question_id": "express_003",
        "repository": "expressjs/express",
        "category": "function_identification",
        "query_style": "identifier",
        "question": "Where is the View constructor defined in Express and what properties does it store?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query locating View constructor.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/view.js",
                "symbol": "function View",
                "start_line": 52,
                "end_line": 80,
                "relevance_reason": "Defines the View constructor responsible for template engine lookup, default engine assignment, and root resolution."
            }
        ]
    },
    {
        "question_id": "express_004",
        "repository": "expressjs/express",
        "category": "function_identification",
        "query_style": "identifier",
        "question": "Where is the req.get (or req.header) method defined and how does it retrieve HTTP request headers?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query locating header retrieval implementation.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/request.js",
                "symbol": "req.header",
                "start_line": 63,
                "end_line": 85,
                "relevance_reason": "Implements case-insensitive request header retrieval with special handling for Referer / Referrer."
            }
        ]
    },
    {
        "question_id": "express_005",
        "repository": "expressjs/express",
        "category": "implementation_understanding",
        "query_style": "conceptual",
        "question": "How does Express register middleware and mount nested sub-applications under a specific route prefix?",
        "conversation_id": "express_conv_02",
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual query describing middleware and sub-app mounting.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/application.js",
                "symbol": "app.use",
                "start_line": 190,
                "end_line": 244,
                "relevance_reason": "Handles flattening of middleware arguments, sub-application mounting with mountpath, and delegating to router.use."
            }
        ]
    },
    {
        "question_id": "express_006",
        "repository": "expressjs/express",
        "category": "cross_file_dependencies",
        "query_style": "conversational",
        "question": "How does that instance lazily initialize its underlying router with case-sensitive and strict routing settings?",
        "conversation_id": "express_conv_02",
        "turn_number": 2,
        "previous_context": "In express_005, we saw how app.use retrieves this.router.",
        "expected_standalone_query": "How does the Express application lazily initialize its router with case-sensitive and strict routing settings?",
        "notes": "Context-dependent query with multi-file evidence across application.js and express.js.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/application.js",
                "symbol": "app.init",
                "start_line": 59,
                "end_line": 83,
                "relevance_reason": "Defines the lazy router property getter with caseSensitive and strict routing settings inside app.init."
            },
            {
                "file_path": "lib/express.js",
                "symbol": "exports.Router = Router",
                "start_line": 68,
                "end_line": 73,
                "relevance_reason": "Exposes the Router constructor on the Express exports object."
            }
        ]
    },
    {
        "question_id": "express_007",
        "repository": "expressjs/express",
        "category": "implementation_understanding",
        "query_style": "conceptual",
        "question": "How does the framework serialize objects, apply indentation settings, and send formatted JSON responses?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual query on JSON serialization and response headers.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/response.js",
                "symbol": "res.json",
                "start_line": 236,
                "end_line": 260,
                "relevance_reason": "Implements res.json applying json replacer, json spaces settings, Content-Type, and calling res.send."
            }
        ]
    },
    {
        "question_id": "express_008",
        "repository": "expressjs/express",
        "category": "implementation_understanding",
        "query_style": "conceptual",
        "question": "How does Express stream static files from disk to the client while handling range headers and transmission errors?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual query on streaming files from disk.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/response.js",
                "symbol": "res.sendFile",
                "start_line": 375,
                "end_line": 425,
                "relevance_reason": "Implements res.sendFile delegating to send-stream with callback and event handling."
            }
        ]
    },
    {
        "question_id": "express_009",
        "repository": "expressjs/express",
        "category": "cross_file_dependencies",
        "query_style": "conceptual",
        "question": "How does the response transmission mechanism calculate and set entity tags using utility hashing generators?",
        "conversation_id": "express_conv_03",
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual multi-file query connecting response.js and utils.js without leaking file paths.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/response.js",
                "symbol": "res.send",
                "start_line": 126,
                "end_line": 190,
                "relevance_reason": "Shows response body conversion, Content-Type setting, and calling app.get('etag fn') to generate ETag headers."
            },
            {
                "file_path": "lib/utils.js",
                "symbol": "exports.compileETag",
                "start_line": 130,
                "end_line": 152,
                "relevance_reason": "Compiles the etag setting to an etag generator function (weak or strong)."
            }
        ]
    },
    {
        "question_id": "express_010",
        "repository": "expressjs/express",
        "category": "architecture_flow",
        "query_style": "conversational",
        "question": "Walk through the dispatch lifecycle when a request first enters that application handler before response generation.",
        "conversation_id": "express_conv_03",
        "turn_number": 2,
        "previous_context": "In express_009, we saw how res.send formats and dispatches the response.",
        "expected_standalone_query": "Walk through the request dispatch lifecycle in app.handle when an HTTP request enters Express.",
        "notes": "Context-dependent query with multi-file evidence across application.js and express.js.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/application.js",
                "symbol": "app.handle",
                "start_line": 152,
                "end_line": 179,
                "relevance_reason": "Shows finalhandler setup, prototype assignment for req and res, and calling this.router.handle."
            },
            {
                "file_path": "lib/express.js",
                "symbol": "function createApplication",
                "start_line": 36,
                "end_line": 45,
                "relevance_reason": "Shows how createApplication sets up app(req, res, next) calling app.handle(req, res, next)."
            }
        ]
    },
    {
        "question_id": "express_011",
        "repository": "expressjs/express",
        "category": "cross_file_dependencies",
        "query_style": "conceptual",
        "question": "How does the template rendering coordinator locate view files on disk and invoke the underlying view renderer?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual multi-file query connecting application.js and view.js.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/application.js",
                "symbol": "app.render",
                "start_line": 522,
                "end_line": 575,
                "relevance_reason": "Shows view lookup from cache, instantiating new View, and invoking view.render with options and callback."
            },
            {
                "file_path": "lib/view.js",
                "symbol": "View.prototype.render",
                "start_line": 133,
                "end_line": 165,
                "relevance_reason": "Implements View.prototype.render invoking the registered template engine on the resolved file path."
            }
        ]
    },
    {
        "question_id": "express_012",
        "repository": "expressjs/express",
        "category": "architecture_flow",
        "query_style": "identifier",
        "question": "How does res.render render a view and send the rendered HTML response back to the client?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query on res.render lifecycle.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/response.js",
                "symbol": "res.render",
                "start_line": 897,
                "end_line": 935,
                "relevance_reason": "Shows merging options with res.locals, delegating to app.render, and sending the rendered string."
            }
        ]
    },
    {
        "question_id": "express_013",
        "repository": "expressjs/express",
        "category": "architecture_flow",
        "query_style": "identifier",
        "question": "How does Express handle HTTP method routing when methods like app.get, app.post, and app.all are invoked?",
        "conversation_id": "express_conv_04",
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Root query on HTTP method routing loop.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/application.js",
                "symbol": "app.all",
                "start_line": 470,
                "end_line": 505,
                "relevance_reason": "Shows loop iterating over HTTP methods and delegating route handler calls to this.route(path)."
            }
        ]
    },
    {
        "question_id": "express_014",
        "repository": "expressjs/express",
        "category": "debugging_configuration",
        "query_style": "conversational",
        "question": "How does it compile and apply proxy settings to resolve client IP addresses for those routes?",
        "conversation_id": "express_conv_04",
        "turn_number": 2,
        "previous_context": "In express_013, we examined HTTP method routing in Express.",
        "expected_standalone_query": "How does Express compile and apply trust proxy configuration to resolve req.ip and req.ips?",
        "notes": "Context-dependent query with multi-file evidence connecting utils.js and request.js.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/utils.js",
                "symbol": "exports.compileTrust",
                "start_line": 194,
                "end_line": 215,
                "relevance_reason": "Compiles boolean, number, string, or function trust proxy configurations using proxy-addr."
            },
            {
                "file_path": "lib/request.js",
                "symbol": "defineGetter(req, 'ip'",
                "start_line": 340,
                "end_line": 355,
                "relevance_reason": "Defines the getter for req.ip calling proxyaddr with the application's trust proxy function."
            }
        ]
    },
    {
        "question_id": "express_015",
        "repository": "expressjs/express",
        "category": "debugging_configuration",
        "query_style": "conceptual",
        "question": "How does Express calculate weak and strong entity tags for caching HTTP responses?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual query on ETag generation without leaking file path.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/utils.js",
                "symbol": "exports.compileETag",
                "start_line": 130,
                "end_line": 152,
                "relevance_reason": "Shows mapping string/boolean ETag settings to weak or strong ETag generator functions."
            }
        ]
    },
    {
        "question_id": "express_016",
        "repository": "expressjs/express",
        "category": "cross_file_dependencies",
        "query_style": "conceptual",
        "question": "How does Express determine client IP addresses through proxy hops based on default configuration settings?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual multi-file query connecting request.js and application.js.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/request.js",
                "symbol": "defineGetter(req, 'ip'",
                "start_line": 340,
                "end_line": 355,
                "relevance_reason": "Defines the getter for req.ip calling proxyaddr with the application's trust proxy function."
            },
            {
                "file_path": "lib/application.js",
                "symbol": "trust proxy",
                "start_line": 99,
                "end_line": 115,
                "relevance_reason": "Sets default trust proxy configuration and handles inheritance on sub-application mount."
            }
        ]
    },
    {
        "question_id": "express_017",
        "repository": "expressjs/express",
        "category": "function_identification",
        "query_style": "identifier",
        "question": "Where is res.cookie defined and how does it serialize cookies and apply options like maxAge and signed?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query locating res.cookie.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/response.js",
                "symbol": "res.cookie",
                "start_line": 748,
                "end_line": 785,
                "relevance_reason": "Shows cookie serialization, maxAge calculation into expires, and appending Set-Cookie headers."
            }
        ]
    },
    {
        "question_id": "express_018",
        "repository": "expressjs/express",
        "category": "debugging_configuration",
        "query_style": "conceptual",
        "question": "How does Express register custom template engines to render non-standard file extensions?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual query on template engine registration without naming app.engine.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "lib/application.js",
                "symbol": "app.engine",
                "start_line": 294,
                "end_line": 309,
                "relevance_reason": "Validates engine callback and registers it in this.engines dictionary keyed by file extension."
            }
        ]
    },
    {
        "question_id": "express_019",
        "repository": "expressjs/express",
        "category": "unanswerable",
        "query_style": "unanswerable",
        "question": "Where is the built-in JWT (JSON Web Token) authentication verification middleware implemented in Express?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Unanswerable: Express does not provide built-in JWT verification middleware.",
        "answerable": False,
        "evidence": [],
        "justification": "Express does not provide built-in JWT verification or authentication middleware. Token-based authentication in Express is handled by external npm packages such as jsonwebtoken or express-jwt."
    },
    {
        "question_id": "express_020",
        "repository": "expressjs/express",
        "category": "unanswerable",
        "query_style": "unanswerable",
        "question": "How does Express configure automatic database migration and schema synchronization for MongoDB?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Unanswerable: Express is strictly an HTTP routing library and has no database migration tools.",
        "answerable": False,
        "evidence": [],
        "justification": "Express is strictly an HTTP routing and middleware framework and does not include database drivers, ORMs, or database migration utilities. MongoDB persistence and migrations are managed via external ODM libraries like Mongoose."
    },

    # =========================================================================
    # FMTLIB/FMT (20 questions: fmt_001 to fmt_020)
    # =========================================================================
    {
        "question_id": "fmt_001",
        "repository": "fmtlib/fmt",
        "category": "function_identification",
        "query_style": "identifier",
        "question": "Where is the text_style class defined in fmt and what style attributes does it store?",
        "conversation_id": "fmt_conv_01",
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query locating text_style (file path stripped).",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/color.h",
                "symbol": "class text_style",
                "start_line": 236,
                "end_line": 275,
                "relevance_reason": "Defines the text_style class representing foreground/background colors and text emphasis (bold, italic, etc.)."
            }
        ]
    },
    {
        "question_id": "fmt_002",
        "repository": "fmtlib/fmt",
        "category": "implementation_understanding",
        "query_style": "conversational",
        "question": "How does the styled helper function apply that style to an argument value for formatted output?",
        "conversation_id": "fmt_conv_01",
        "turn_number": 2,
        "previous_context": "In fmt_001, we inspected text_style and terminal color representations.",
        "expected_standalone_query": "How does the styled helper function apply a text_style to an argument value for formatted output in fmt?",
        "notes": "Context-dependent conversational query using 'that style'.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/color.h",
                "symbol": "styled",
                "start_line": 660,
                "end_line": 673,
                "relevance_reason": "Defines the styled template function wrapping an argument value and a text_style into a styled_arg."
            }
        ]
    },
    {
        "question_id": "fmt_003",
        "repository": "fmtlib/fmt",
        "category": "function_identification",
        "query_style": "conceptual",
        "question": "How does the format library provide RAII management for C-style file streams?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual query describing RAII file handle wrapping without naming buffered_file or os.h.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/os.h",
                "symbol": "class buffered_file",
                "start_line": 167,
                "end_line": 205,
                "relevance_reason": "Defines buffered_file RAII class wrapping FILE* with move semantics and automated closure."
            }
        ]
    },
    {
        "question_id": "fmt_004",
        "repository": "fmtlib/fmt",
        "category": "function_identification",
        "query_style": "conceptual",
        "question": "How does fmt construct dynamic argument lists when format arguments are not known at compile time?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual query on dynamic formatting arguments without naming dynamic_format_arg_store or args.h.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/args.h",
                "symbol": "class dynamic_format_arg_store",
                "start_line": 74,
                "end_line": 110,
                "relevance_reason": "Defines dynamic_format_arg_store for storing dynamically sized lists of formatting arguments at runtime."
            }
        ]
    },
    {
        "question_id": "fmt_005",
        "repository": "fmtlib/fmt",
        "category": "implementation_understanding",
        "query_style": "identifier",
        "question": "How does fmt detect whether a type satisfies range requirements via the is_range trait?",
        "conversation_id": "fmt_conv_02",
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query on range type trait (file path stripped).",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/ranges.h",
                "symbol": "struct is_range",
                "start_line": 370,
                "end_line": 410,
                "relevance_reason": "Implements compile-time detection of begin() and end() iterators to determine range support."
            }
        ]
    },
    {
        "question_id": "fmt_006",
        "repository": "fmtlib/fmt",
        "category": "cross_file_dependencies",
        "query_style": "conversational",
        "question": "How does join construct a view to format those detected ranges with custom delimiters?",
        "conversation_id": "fmt_conv_02",
        "turn_number": 2,
        "previous_context": "In fmt_005, we examined the is_range trait for range detection.",
        "expected_standalone_query": "How does fmt::join construct a join_view to format ranges with custom delimiters?",
        "notes": "Context-dependent query using 'those detected ranges' with multi-span evidence.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/ranges.h",
                "symbol": "join",
                "start_line": 865,
                "end_line": 890,
                "relevance_reason": "Defines the join function returning a join_view over the range and separator."
            },
            {
                "file_path": "include/fmt/ranges.h",
                "symbol": "struct join_view",
                "start_line": 670,
                "end_line": 705,
                "relevance_reason": "Defines join_view struct that iterates elements and formats custom delimiters."
            }
        ]
    },
    {
        "question_id": "fmt_007",
        "repository": "fmtlib/fmt",
        "category": "implementation_understanding",
        "query_style": "identifier",
        "question": "How does dynamic_format_arg_store::push_back store typed values and named arguments?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query on push_back method (file path stripped).",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/args.h",
                "symbol": "push_back",
                "start_line": 153,
                "end_line": 195,
                "relevance_reason": "Shows storing regular values and named arguments into dynamic_arg_list."
            }
        ]
    },
    {
        "question_id": "fmt_008",
        "repository": "fmtlib/fmt",
        "category": "cross_file_dependencies",
        "query_style": "identifier",
        "question": "How does the ostream class in fmt provide buffered output for low-level file descriptors?",
        "conversation_id": "fmt_conv_03",
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Root query on ostream buffer with multi-span evidence.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/os.h",
                "symbol": "class ostream",
                "start_line": 361,
                "end_line": 400,
                "relevance_reason": "Implements buffered write operations on file descriptors extending detail::buffer<char>."
            },
            {
                "file_path": "include/fmt/os.h",
                "symbol": "class buffered_file",
                "start_line": 167,
                "end_line": 205,
                "relevance_reason": "Manages file descriptor attachment and lifecycle via fdopen and buffered_file."
            }
        ]
    },
    {
        "question_id": "fmt_009",
        "repository": "fmtlib/fmt",
        "category": "architecture_flow",
        "query_style": "conversational",
        "question": "Walk through the execution flow when printing styled text to that buffered stream.",
        "conversation_id": "fmt_conv_03",
        "turn_number": 2,
        "previous_context": "In fmt_008, we examined buffered output in fmt.",
        "expected_standalone_query": "Walk through the execution flow of vprint when formatting colored text to a file stream in fmt.",
        "notes": "Context-dependent query using 'that buffered stream' with multi-file evidence across color.h and text_style.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/color.h",
                "symbol": "vprint",
                "start_line": 497,
                "end_line": 520,
                "relevance_reason": "Shows buffer creation, applying ANSI escape sequences for text_style, formatting arguments, and writing to FILE*."
            },
            {
                "file_path": "include/fmt/color.h",
                "symbol": "class text_style",
                "start_line": 236,
                "end_line": 275,
                "relevance_reason": "Supplies color specifications and emphasis flags evaluated during vprint."
            }
        ]
    },
    {
        "question_id": "fmt_010",
        "repository": "fmtlib/fmt",
        "category": "function_identification",
        "query_style": "identifier",
        "question": "Where is the printf_arg_formatter class defined and what format specifiers does it parse?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query locating printf_arg_formatter (file path stripped).",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/printf.h",
                "symbol": "class printf_arg_formatter",
                "start_line": 232,
                "end_line": 270,
                "relevance_reason": "Defines printf_arg_formatter subclassing arg_formatter to handle POSIX printf specifications."
            }
        ]
    },
    {
        "question_id": "fmt_011",
        "repository": "fmtlib/fmt",
        "category": "cross_file_dependencies",
        "query_style": "identifier",
        "question": "How does fmt::sprintf invoke vsprintf and detail::vprintf to produce a formatted string?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query with multi-span evidence connecting sprintf and printf_arg_formatter.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/printf.h",
                "symbol": "sprintf",
                "start_line": 604,
                "end_line": 625,
                "relevance_reason": "Shows construction of memory_buffer and dispatch to vprintf with printf arguments."
            },
            {
                "file_path": "include/fmt/printf.h",
                "symbol": "class printf_arg_formatter",
                "start_line": 232,
                "end_line": 270,
                "relevance_reason": "Provides the argument formatter visitor used during vprintf formatting."
            }
        ]
    },
    {
        "question_id": "fmt_012",
        "repository": "fmtlib/fmt",
        "category": "architecture_flow",
        "query_style": "conceptual",
        "question": "How does fmt format std::tuple and std::pair objects into delimiter-separated elements?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual query on tuple formatting without naming tuple_join_view or ranges.h.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/ranges.h",
                "symbol": "tuple_join_view",
                "start_line": 722,
                "end_line": 765,
                "relevance_reason": "Implements tuple unpacking and formatting with custom delimiters."
            }
        ]
    },
    {
        "question_id": "fmt_013",
        "repository": "fmtlib/fmt",
        "category": "debugging_configuration",
        "query_style": "conceptual",
        "question": "How does fmt parse and format time durations with unit suffixes like seconds and milliseconds?",
        "conversation_id": "fmt_conv_04",
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual root query on chrono duration formatting.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/chrono.h",
                "symbol": "formatter<std::chrono::duration",
                "start_line": 2088,
                "end_line": 2130,
                "relevance_reason": "Defines formatter for std::chrono::duration handling unit suffixes and precision."
            }
        ]
    },
    {
        "question_id": "fmt_014",
        "repository": "fmtlib/fmt",
        "category": "debugging_configuration",
        "query_style": "conversational",
        "question": "How can that formatting syntax be parsed at compile time using macros?",
        "conversation_id": "fmt_conv_04",
        "turn_number": 2,
        "previous_context": "In fmt_013, we saw chrono duration formatting in fmt.",
        "expected_standalone_query": "How does fmt define FMT_COMPILE to enable compile-time format string parsing?",
        "notes": "Context-dependent query using 'that formatting syntax'.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/compile.h",
                "symbol": "FMT_COMPILE",
                "start_line": 35,
                "end_line": 55,
                "relevance_reason": "Defines FMT_COMPILE macro wrapping string literals in compiled_string for static analysis."
            }
        ]
    },
    {
        "question_id": "fmt_015",
        "repository": "fmtlib/fmt",
        "category": "architecture_flow",
        "query_style": "conceptual",
        "question": "How does fmt construct and report Windows operating system errors from native error codes?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual query on Windows error translation with multi-span evidence.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/os.h",
                "symbol": "windows_error",
                "start_line": 140,
                "end_line": 165,
                "relevance_reason": "Implements windows_error constructing std::system_error from GetLastError() codes."
            },
            {
                "file_path": "include/fmt/os.h",
                "symbol": "report_windows_error",
                "start_line": 157,
                "end_line": 165,
                "relevance_reason": "Reports Windows errors safely without throwing exceptions for destructor contexts."
            }
        ]
    },
    {
        "question_id": "fmt_016",
        "repository": "fmtlib/fmt",
        "category": "implementation_understanding",
        "query_style": "conceptual",
        "question": "How does fmt convert C++ enum values to their numeric representations during formatting?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Conceptual query describing enum conversion without naming format_as or enum.h.",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/enum.h",
                "symbol": "format_as",
                "start_line": 245,
                "end_line": 267,
                "relevance_reason": "Defines format_as returning static_cast<underlying_t<E>>(value) for enum formatting."
            }
        ]
    },
    {
        "question_id": "fmt_017",
        "repository": "fmtlib/fmt",
        "category": "debugging_configuration",
        "query_style": "identifier",
        "question": "How does fmt implement as_underlying to force formatting an enum as an integer?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query on as_underlying (file path stripped).",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/enum.h",
                "symbol": "as_underlying",
                "start_line": 50,
                "end_line": 75,
                "relevance_reason": "Defines as_underlying marker struct and constant for explicit integer representation."
            }
        ]
    },
    {
        "question_id": "fmt_018",
        "repository": "fmtlib/fmt",
        "category": "debugging_configuration",
        "query_style": "identifier",
        "question": "How does fmt define terminal color codes in the terminal_color enum class?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Identifier query on terminal_color (file path stripped).",
        "answerable": True,
        "evidence": [
            {
                "file_path": "include/fmt/color.h",
                "symbol": "terminal_color",
                "start_line": 160,
                "end_line": 200,
                "relevance_reason": "Defines standard 16-color ANSI terminal color codes (black, red, green, etc.)."
            }
        ]
    },
    {
        "question_id": "fmt_019",
        "repository": "fmtlib/fmt",
        "category": "unanswerable",
        "query_style": "unanswerable",
        "question": "Where is the built-in asynchronous non-blocking thread-pool logging sink configured in fmt?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Unanswerable: fmt is a formatting library, not an async logging engine.",
        "answerable": False,
        "evidence": [],
        "justification": "fmt is a text formatting library, not an asynchronous logging framework. Thread-pooled asynchronous logging sinks are provided by libraries built on top of fmt (such as spdlog or quill), not by fmtlib itself."
    },
    {
        "question_id": "fmt_020",
        "repository": "fmtlib/fmt",
        "category": "unanswerable",
        "query_style": "unanswerable",
        "question": "How does fmt parse and serialize JSON documents into C++ struct fields?",
        "conversation_id": None,
        "turn_number": 1,
        "previous_context": None,
        "expected_standalone_query": None,
        "notes": "Unanswerable: fmt does not include a JSON parser or serializer.",
        "answerable": False,
        "evidence": [],
        "justification": "fmt is strictly a format string interpolation and rendering library; it does not contain a JSON parser or serializer (such as nlohmann/json or rapidjson)."
    }
]

def generate():
    eval_questions = []
    ground_truth = []
    multi_evidence_count = 0

    for item in UPDATED_BENCHMARK:
        qid = item["question_id"]
        eval_questions.append({
            "question_id": qid,
            "repository": item["repository"],
            "category": item["category"],
            "query_style": item["query_style"],
            "question": item["question"],
            "conversation_id": item["conversation_id"],
            "turn_number": item["turn_number"],
            "previous_context": item["previous_context"],
            "expected_standalone_query": item["expected_standalone_query"],
            "notes": item["notes"]
        })

        gt_item = {
            "question_id": qid,
            "repository": item["repository"],
            "answerable": item["answerable"]
        }
        if item["answerable"]:
            gt_item["evidence"] = item["evidence"]
            if len(item["evidence"]) >= 2:
                multi_evidence_count += 1
        else:
            gt_item["evidence"] = []
            gt_item["justification"] = item["justification"]

        ground_truth.append(gt_item)

    # Save evaluation_questions.json
    eq_path = os.path.join(DATASETS_DIR, "evaluation_questions.json")
    with open(eq_path, "w", encoding="utf-8") as f:
        json.dump(eval_questions, f, indent=2)

    # Save ground_truth.json
    gt_path = os.path.join(DATASETS_DIR, "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated {len(eval_questions)} evaluation questions in {eq_path}")
    print(f"Generated {len(ground_truth)} ground truth records ({multi_evidence_count} multi-evidence questions) in {gt_path}")

if __name__ == "__main__":
    generate()
